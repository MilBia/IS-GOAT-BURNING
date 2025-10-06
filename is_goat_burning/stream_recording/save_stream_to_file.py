"""Handles the saving of the video stream to disk in timed chunks.

This module provides the `AsyncVideoChunkSaver` class, which is responsible for
asynchronously writing video frames to MP4 files. It manages file rotation,
enforces disk space limits, and contains special logic for archiving video
chunks before and after a fire event is detected.
"""

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from functools import partial
import os
import shutil
import time
from typing import Any
from typing import ClassVar

import cv2
import numpy as np

from is_goat_burning.config import settings
from is_goat_burning.fire_detection.signal_handler import SignalHandler
from is_goat_burning.logger import get_logger

logger = get_logger("AsyncVideoChunkSaver")


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class AsyncVideoChunkSaver:
    """Manages saving a video stream into timed, rotating file chunks.

    This class provides an asynchronous interface to save video frames. It supports
    two modes, configured via `settings.video.buffer_mode`:
    - "disk": Uses a background task to continuously write frames to disk in
      chunks.
    - "memory": Holds frames in a rolling in-memory buffer (`collections.deque`)
      and only writes to disk when a fire event is triggered.

    It integrates with the `SignalHandler` to perform special archiving logic
    when a fire is detected.

    Attributes:
        enabled (bool): If False, all operations are no-ops.
        output_dir (str): The directory where video chunks are saved.
        chunk_length_seconds (int): The duration of each video chunk file.
        max_chunks (int): The maximum number of non-archived chunks to keep on
            disk. Older chunks are deleted to enforce this limit.
        chunks_to_keep_after_fire (int): The number of additional chunks to
            record and archive after a fire is first detected.
        fps (float): The frames per second of the video stream.
        frame_queue (asyncio.Queue): A queue for incoming frames to be written
            (used in "disk" mode and post-fire "memory" mode).
        archive_queue (asyncio.Queue): A queue for video chunks to be archived.
        writer (cv2.VideoWriter): The OpenCV video writer instance.
        pre_fire_buffer (deque): A deque that stores the paths of the most
            recent video chunks, used for pre-event archiving ("disk" mode).
        memory_buffer (deque): A deque that stores recent frames in memory
            (used in "memory" mode).
    """

    # --- Configuration ---
    enabled: bool
    output_dir: str
    chunk_length_seconds: int
    max_chunks: int
    chunks_to_keep_after_fire: int
    fps: float = 30.0
    FILENAME_PREFIX: ClassVar[str] = "goat-cam_"
    FILENAME_SUFFIX: ClassVar[str] = ".mp4"
    VIDEO_CODEC: ClassVar[str] = "mp4v"
    MAX_TIMEOUT_RETRIES: ClassVar[int] = 3

    # --- Internal State ---
    frame_queue: asyncio.Queue[np.ndarray | None] = field(init=False, default_factory=asyncio.Queue)
    archive_queue: asyncio.Queue[tuple[str, str] | None] = field(init=False, default_factory=asyncio.Queue)
    writer: cv2.VideoWriter | None = field(init=False, default=None, repr=False)
    chunk_start_time: float = field(init=False, default=0.0)
    current_video_path: str | None = field(init=False, default=None)
    _main_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _archive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    __call__: Callable[[np.ndarray], None]
    chunk_limit_action: Callable[[], None] = field(init=False, repr=False)
    makedirs_callable: Callable[..., None] = field(init=False, repr=False)
    signal_handler: SignalHandler = field(init=False, default_factory=SignalHandler)
    pre_fire_buffer: deque[str] = field(init=False)
    memory_buffer: deque[np.ndarray] | None = field(init=False, default=None)
    is_new_chunk: bool = field(init=False, default=False)
    _fire_handling_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        """Initializes state and methods based on the `enabled` flag and buffer mode."""
        self.pre_fire_buffer = deque(maxlen=self.max_chunks)
        self.__call__ = self._noop
        self.chunk_limit_action = self._noop
        self.makedirs_callable = partial(os.makedirs, exist_ok=True)
        if not self.enabled:
            return

        self.create_storage_directory()
        buffer_mode = settings.video.buffer_mode
        if buffer_mode == "memory":
            logger.info("Video saver initializing in MEMORY buffer mode.")
            buffer_size = int(settings.video.memory_buffer_seconds * self.fps)
            self.memory_buffer = deque(maxlen=buffer_size)
            self.__call__ = self._add_frame_to_memory_buffer
        elif buffer_mode == "disk":
            logger.info("Video saver initializing in DISK buffer mode.")
            self.__call__ = self._queue_frame
            if self.max_chunks > 0:
                self.chunk_limit_action = self._enforce_chunk_limit_blocking
        else:
            raise NotImplementedError(f"Buffer mode '{buffer_mode}' is not implemented.")

    def create_storage_directory(self) -> None:
        """Creates the output directory if it doesn't exist."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Video chunks will be saved to: {self.output_dir}")
        except OSError as e:
            logger.error(f"Could not create directory {self.output_dir}: {e}")
            self.__call__ = self._noop  # Disable if directory creation fails

    def start(self) -> None:
        """Starts the necessary background tasks based on the operating mode."""
        if not self.enabled:
            return

        self._archive_task = asyncio.create_task(self.archive_task())
        self._main_task = asyncio.create_task(self._run_main_loop())

    async def stop(self) -> None:
        """Signals the background tasks to stop and waits for them to finish."""
        if not self.enabled:
            return

        tasks = []
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            tasks.append(self._main_task)

        if self._archive_task and not self._archive_task.done():
            # The archive task waits on a queue, so putting None is a clean
            # way to make it exit its loop before we gather it.
            await self.archive_queue.put(None)
            tasks.append(self._archive_task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _noop(self, *args: Any, **kwargs: Any) -> None:
        """A "no-operation" method used when the saver is disabled."""
        pass

    def _enforce_chunk_limit_blocking(self) -> None:
        """Deletes the oldest video chunk(s) if the `max_chunks` limit is exceeded."""
        try:
            files = sorted(
                f
                for f in os.listdir(self.output_dir)
                if f.startswith(self.FILENAME_PREFIX)
                and f.endswith(self.FILENAME_SUFFIX)
                and os.path.isfile(os.path.join(self.output_dir, f))
            )
            num_to_delete = len(files) - self.max_chunks + 1
            if num_to_delete > 0:
                for oldest_file in files[:num_to_delete]:
                    file_path_to_delete = os.path.join(self.output_dir, oldest_file)
                    try:
                        os.remove(file_path_to_delete)
                        logger.debug(f"Removed oldest chunk: {oldest_file}")
                    except OSError as e:
                        logger.error(f"Could not remove file {file_path_to_delete}: {e}")
        except OSError as e:
            logger.error(f"Error listing files in {self.output_dir} for cleanup: {e}")

    def _start_new_chunk_blocking(self, frame_size: tuple[int, int]) -> str | None:
        """Closes the current video chunk and starts a new one.

        This is a synchronous method intended to be run in an executor.

        Args:
            frame_size: A tuple of (width, height) for the new video chunk.

        Returns:
            The file path of the chunk that was just closed, or None if there
            was no previous chunk.
        """
        self.chunk_limit_action()

        previous_chunk_path = self.current_video_path
        if self.writer is not None:
            self.writer.release()
            if previous_chunk_path:
                logger.info(f"Saved video chunk: {os.path.basename(previous_chunk_path)}")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_video_path = os.path.join(self.output_dir, f"{self.FILENAME_PREFIX}{timestamp}{self.FILENAME_SUFFIX}")
        fourcc = cv2.VideoWriter_fourcc(*self.VIDEO_CODEC)
        self.writer = cv2.VideoWriter(self.current_video_path, fourcc, self.fps, frame_size)
        self.chunk_start_time = time.time()
        self.is_new_chunk = True
        logger.info(f"Started new video chunk: {os.path.basename(self.current_video_path)}")
        return previous_chunk_path

    def _write_frame_blocking(self, frame: np.ndarray) -> str | None:
        """Writes a single frame to the current video chunk.

        This method handles the logic for starting a new chunk when the time
        limit is reached. It is a synchronous method intended to be run in an
        executor.

        Args:
            frame: The video frame (as a NumPy array) to be written.

        Returns:
            The path of a video chunk if it was just closed, otherwise None.
        """
        closed_chunk_path: str | None = None
        is_new_chunk_needed = self.writer is None or (time.time() - self.chunk_start_time) >= self.chunk_length_seconds

        if is_new_chunk_needed:
            height, width, _ = frame.shape
            closed_chunk_path = self._start_new_chunk_blocking(frame_size=(width, height))

        if self.writer:
            self.writer.write(frame)
            if self.is_new_chunk:
                self.pre_fire_buffer.append(self.current_video_path)
                self.is_new_chunk = False
        return closed_chunk_path

    def _queue_frame(self, frame: np.ndarray) -> None:
        """Puts a frame onto the asynchronous queue to be written."""
        try:
            self.frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Frame queue is full, dropping a frame.")

    def _add_frame_to_memory_buffer(self, frame: np.ndarray) -> None:
        """Adds a frame to the in-memory buffer deque."""
        if self.memory_buffer is not None:
            self.memory_buffer.append(frame)

    def reset_after_fire(self) -> None:
        """Resets the state after handling a fire event."""
        self.pre_fire_buffer.clear()
        self.signal_handler.reset_fire_event()
        # If in memory mode, switch back to buffering in memory
        if settings.video.buffer_mode == "memory":
            self.__call__ = self._add_frame_to_memory_buffer
        logger.debug("FIRE EVENT HANDLING COMPLETE. Resuming normal operations.")

    async def _run_main_loop(self) -> None:
        """Starts the appropriate main loop based on the buffer mode."""
        try:
            if settings.video.buffer_mode == "disk":
                await self._writer_task()
            elif settings.video.buffer_mode == "memory":
                await self._memory_mode_event_loop()
        except asyncio.CancelledError:
            logger.info("Main saver task was cancelled.")
            # Re-raise to ensure the task actually terminates.
            raise
        finally:
            if self.writer is not None:
                self.writer.release()
                if self.current_video_path:
                    logger.info(f"Saved final video chunk on exit: {self.current_video_path}")

    async def _memory_mode_event_loop(self) -> None:
        """The main event loop when operating in 'memory' buffer mode.

        This loop waits efficiently for the fire signal and triggers the
        event handling logic when it is set.
        """
        while True:
            # This will suspend the task indefinitely until the event is set.
            await self.signal_handler.fire_detected_event.wait()

            # Once the event is set, we proceed to handle it.
            await self._handle_fire_event_async()
            self.reset_after_fire()

    async def _writer_task(self) -> None:
        """The main consumer task that writes frames from the queue to disk."""
        loop = asyncio.get_running_loop()
        while True:
            if self.signal_handler.is_fire_detected() and not self._fire_handling_lock.locked():
                async with self._fire_handling_lock:
                    if self.signal_handler.is_fire_detected():
                        try:
                            await self._handle_fire_event_async()
                        finally:
                            self.reset_after_fire()

            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1)
                if frame is None:
                    break
                await loop.run_in_executor(None, self._write_frame_blocking, frame)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("Writer task cancelled during frame wait.")
                raise  # Ensure cancellation propagates
            except Exception:
                logger.exception("Error in writer task, but will continue running.")

    def __call__(self, frame: np.ndarray) -> None:
        """Allows the class instance to be called like a function.

        This method is dynamically assigned during initialization based on the
        configured buffer mode.
        """
        raise NotImplementedError("This class must be initialized before it can be called.")

    def _flush_buffer_to_disk_blocking(self, frames: deque[np.ndarray], path: str) -> None:
        """Writes a deque of frames to a single video file.

        This is a synchronous, blocking method designed to be run in an executor.

        Args:
            frames: A deque of video frames to write.
            path: The full path of the video file to create.
        """
        if not frames:
            logger.warning("Memory buffer is empty, cannot flush to disk.")
            return

        # Get frame size from the first frame
        first_frame = frames[0]
        height, width, _ = first_frame.shape
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*self.VIDEO_CODEC)
        writer = cv2.VideoWriter(path, fourcc, self.fps, frame_size)
        if not writer.isOpened():
            logger.error(f"Failed to open video writer for path: {path}")
            return

        logger.info(f"Flushing {len(frames)} frames from memory to {os.path.basename(path)}...")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
            logger.info(f"Finished flushing buffer to {os.path.basename(path)}.")

    @staticmethod
    def _move_file_blocking(chunk_path: str, event_dir: str) -> None:
        """Moves a file from the main directory to an event archive directory."""
        try:
            if os.path.exists(chunk_path):
                shutil.move(chunk_path, event_dir)
                logger.debug(f"Archived {os.path.basename(chunk_path)}.")
            else:
                logger.warning(f"Chunk {chunk_path} not found for archiving.")
        except (OSError, shutil.Error) as e:
            logger.error(f"Could not move chunk {chunk_path}: {e}")

    async def archive_task(self) -> None:
        """A background task that archives files from the `archive_queue`."""
        loop = asyncio.get_running_loop()
        while True:
            item = await self.archive_queue.get()
            if item is None:
                break
            chunk_path, event_dir = item
            await loop.run_in_executor(None, self._move_file_blocking, chunk_path, event_dir)

    async def _create_event_directory(self) -> str | None:
        """Creates a timestamped directory for a fire event.

        Returns:
            The path to the created directory, or None if creation failed.
        """
        event_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        event_dir = os.path.join(self.output_dir, f"event_{event_timestamp}")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.makedirs_callable, event_dir)
            logger.info(f"Created event archive directory: {event_dir}")
            return event_dir
        except OSError as e:
            logger.error(f"Could not create event directory {event_dir}: {e}. Aborting archive.")
            return None

    async def _handle_fire_event_async(self) -> None:
        """Orchestrates the archiving of pre-fire and post-fire chunks."""
        logger.warning("FIRE EVENT TRIGGERED! Archiving will occur in the background.")
        event_dir = await self._create_event_directory()
        if not event_dir:
            return

        original_cleanup_action = self.chunk_limit_action
        self.chunk_limit_action = self._noop

        try:
            if settings.video.buffer_mode == "memory":
                await self._handle_memory_mode_fire_event(event_dir)
            else:  # 'disk' mode
                await self._handle_disk_mode_fire_event(event_dir)
        finally:
            self.chunk_limit_action = original_cleanup_action
            logger.info("Fire event handled. Restoring normal chunk rotation policy.")

    async def _handle_disk_mode_fire_event(self, event_dir: str) -> None:
        """Handles the fire event for 'disk' buffering mode."""
        # Archive all chunks currently in the buffer except the active one.
        cold_chunks = list(self.pre_fire_buffer)[:-1]
        logger.info(f"Queueing {len(cold_chunks)} pre-fire chunks for archiving.")
        for path in cold_chunks:
            try:
                self.archive_queue.put_nowait((path, event_dir))
            except asyncio.QueueFull:
                logger.error(f"Archive queue is full. Failed to queue chunk for archiving {path}.")

        await self._finalize_active_chunk_async(event_dir)
        await self._save_post_fire_chunks_async(event_dir)

    async def _handle_memory_mode_fire_event(self, event_dir: str) -> None:
        """Handles the fire event for 'memory' buffering mode."""
        await self._flush_and_archive_memory_buffer(event_dir)

        # Switch to disk mode to capture post-fire chunks.
        logger.info("Switching to disk mode for post-fire recording.")
        self.__call__ = self._queue_frame

        await self._save_post_fire_chunks_async(event_dir)

    async def _flush_and_archive_memory_buffer(self, event_dir: str) -> None:
        """Flushes the in-memory frame buffer to a file in the event directory."""
        if self.memory_buffer is None:
            logger.error("Memory buffer not initialized, cannot handle fire event.")
            return

        frames_to_flush = self.memory_buffer.copy()
        self.memory_buffer.clear()
        logger.info(f"Captured {len(frames_to_flush)} pre-fire frames from memory.")

        if not frames_to_flush:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pre_fire_path = os.path.join(event_dir, f"pre-fire-buffer_{timestamp}{self.FILENAME_SUFFIX}")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._flush_buffer_to_disk_blocking, frames_to_flush, pre_fire_path)

    async def _finalize_active_chunk_async(self, event_dir: str) -> None:
        """Continues recording until the currently active chunk is finished.

        Once the active chunk (the one being written to when the fire was
        detected) is closed, it is queued for archiving.

        Args:
            event_dir: The destination directory for the archive.
        """
        logger.info("Finalizing the video chunk active during fire detection...")
        loop = asyncio.get_running_loop()
        timeout_retries = 0
        while True:
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=5.0)
                if frame is None:
                    logger.warning("Shutdown signaled while finalizing active chunk.")
                    return
                closed_chunk = await loop.run_in_executor(None, self._write_frame_blocking, frame)
                if closed_chunk:
                    logger.info("Active chunk finalized and queued for archive.")
                    try:
                        self.archive_queue.put_nowait((closed_chunk, event_dir))
                    except asyncio.QueueFull:
                        logger.error(f"Archive queue is full. Failed to queue chunk {closed_chunk}.")
                    return
            except TimeoutError:
                timeout_retries += 1
                logger.warning(f"Timeout waiting for frame... Retry {timeout_retries}/{self.MAX_TIMEOUT_RETRIES}...")
                if timeout_retries >= self.MAX_TIMEOUT_RETRIES:
                    logger.error("Max retries exceeded waiting for frame. Aborting event.")
                    return

    async def _save_post_fire_chunks_async(self, event_dir: str) -> None:
        """Saves a configured number of additional chunks after a fire event.

        Args:
            event_dir: The destination directory for the archive.
        """
        if self.chunks_to_keep_after_fire <= 0:
            return

        logger.info(f"Now saving {self.chunks_to_keep_after_fire} post-fire chunks...")
        loop = asyncio.get_running_loop()
        chunks_saved_count = 0
        timeout_retries = 0
        while chunks_saved_count < self.chunks_to_keep_after_fire:
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=5.0)
                if frame is None:
                    logger.warning("Shutdown signaled during post-fire recording.")
                    return
                closed_chunk = await loop.run_in_executor(None, self._write_frame_blocking, frame)
                if closed_chunk:
                    chunks_saved_count += 1
                    try:
                        self.archive_queue.put_nowait((closed_chunk, event_dir))
                    except asyncio.QueueFull:
                        logger.error(f"Archive queue is full. Failed to queue chunk {closed_chunk}.")
            except TimeoutError:
                timeout_retries += 1
                logger.warning(f"Timeout waiting for frame... Retry {timeout_retries}/{self.MAX_TIMEOUT_RETRIES}...")
                if timeout_retries >= self.MAX_TIMEOUT_RETRIES:
                    logger.error("Max retries exceeded waiting for frame during post-fire recording.")
                    return
