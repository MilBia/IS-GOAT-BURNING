"""Handles the saving of the video stream to disk in timed chunks.

This module provides the `AsyncVideoChunkSaver` class, which is responsible for
asynchronously writing video frames to MP4 files. It manages file rotation,
enforces disk space limits, and contains special logic for archiving video
chunks before and after a fire event is detected.

This class uses a Strategy pattern to delegate the specific buffering logic
(e.g., 'disk' vs 'memory' mode) to dedicated strategy objects.
"""

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from functools import partial
import itertools
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
from is_goat_burning.stream_recording.strategies import BufferStrategy
from is_goat_burning.stream_recording.strategies import DiskBufferStrategy
from is_goat_burning.stream_recording.strategies import MemoryBufferStrategy

logger = get_logger("AsyncVideoChunkSaver")


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class AsyncVideoChunkSaver:
    """Manages saving a video stream into timed, rotating file chunks using a buffer strategy.

    This class acts as the context for a buffer strategy (`DiskBufferStrategy`
    or `MemoryBufferStrategy`), which is chosen based on the application's
    configuration. It provides the shared, low-level functionality for writing
    video files and archiving them, while the strategy object manages the
    high-level logic of when and how frames are processed.

    Attributes:
        enabled (bool): If False, all operations are no-ops.
        output_dir (str): The directory where video chunks are saved.
        chunk_length_seconds (int): The duration of each video chunk file.
        max_chunks (int): The maximum number of non-archived chunks to keep on
            disk. Older chunks are deleted to enforce this limit.
        chunks_to_keep_after_fire: The number of chunk-lengths of video to save after a
            fire is detected.
        fps (float): The frames per second of the video stream.
        frame_queue (asyncio.Queue): A queue for incoming frames to be written
            to disk, primarily used by the DiskBufferStrategy or during
            post-fire recording.
        archive_queue (asyncio.Queue): A queue for video chunk paths that need
            to be moved to an event archive directory.
        writer (cv2.VideoWriter | None): The OpenCV video writer instance for
            the currently active video chunk.
        pre_fire_buffer (deque[str] | None): A deque that stores the paths of the most
            recent video chunks, used for pre-event archiving. Initialized only
            in 'disk' mode.
        strategy (BufferStrategy): The active buffering strategy instance.
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
    FRAME_QUEUE_POLL_TIMEOUT: ClassVar[float] = 1.0
    POST_FIRE_FRAME_TIMEOUT: ClassVar[float] = 5.0

    # --- Internal State ---
    frame_queue: asyncio.Queue[np.ndarray | None] = field(init=False, default_factory=asyncio.Queue)
    archive_queue: asyncio.Queue[tuple[str, str] | None] = field(init=False, default_factory=asyncio.Queue)
    writer: cv2.VideoWriter | None = field(init=False, default=None, repr=False)
    chunk_start_time: float = field(init=False, default=0.0)
    current_video_path: str | None = field(init=False, default=None)
    _archive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    chunk_limit_action: Callable[[], None] = field(init=False, repr=False)
    makedirs_callable: Callable[..., None] = field(init=False, repr=False)
    signal_handler: SignalHandler = field(init=False, default_factory=SignalHandler)
    pre_fire_buffer: deque[str] | None = field(init=False, default=None)
    is_new_chunk: bool = field(init=False, default=False)
    _fire_handling_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    strategy: BufferStrategy = field(init=False)

    def __post_init__(self) -> None:
        """Initializes state and selects the appropriate buffer strategy."""
        self.chunk_limit_action = self._noop
        self.makedirs_callable = partial(os.makedirs, exist_ok=True)
        if not self.enabled:
            return

        self.create_storage_directory()
        buffer_mode = settings.video.buffer_mode
        if buffer_mode == "memory":
            logger.info("Video saver initializing with MEMORY buffer strategy.")
            self.strategy = MemoryBufferStrategy(context=self)
        elif buffer_mode == "disk":
            logger.info("Video saver initializing with DISK buffer strategy.")
            self.strategy = DiskBufferStrategy(context=self)
            self.pre_fire_buffer = deque(maxlen=self.max_chunks)
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
            self.enabled = False

    def start(self) -> None:
        """Starts the background archive task and the selected strategy's main loop."""
        if not self.enabled:
            return

        self._archive_task = asyncio.create_task(self.archive_task())
        asyncio.create_task(self.strategy.start())

    async def stop(self) -> None:
        """Signals the background tasks to stop and waits for them to finish."""
        if not self.enabled:
            return

        tasks = [self.strategy.stop()]

        if self._archive_task and not self._archive_task.done():
            await self.archive_queue.put(None)
            tasks.append(self._archive_task)

        if tasks:
            # Shield tasks to ensure cleanup completes even if stop() is cancelled.
            await asyncio.gather(*(asyncio.shield(t) for t in tasks), return_exceptions=True)

    def _noop(self, *args: Any, **kwargs: Any) -> None:
        """A "no-operation" method used when the saver is disabled."""
        pass

    def _release_writer(self) -> None:
        """Releases the video writer and logs the final saved chunk."""
        if self.writer is not None:
            self.writer.release()
            if self.current_video_path:
                # Log final chunk based on basename to avoid leaking full path
                logger.info(f"Finalized video chunk: {os.path.basename(self.current_video_path)}")
            self.writer = None

    def _drain_frame_queue(self) -> None:
        """Safely drains the frame queue to discard any pending frames."""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

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

    def _start_new_chunk_blocking(self, frame_size: tuple[int, int], target_dir: str | None = None) -> str | None:
        """Closes the current video chunk and starts a new one.

        Args:
            frame_size: A tuple of (width, height) for the new video chunk.
            target_dir: If provided, the new chunk is created in this directory;
                otherwise, it's created in the default output directory.

        Returns:
            The file path of the chunk that was just closed, or None if there
            was no previous chunk.
        """
        if target_dir is None:
            self.chunk_limit_action()

        previous_chunk_path = self.current_video_path
        if self.writer is not None:
            self.writer.release()
            if previous_chunk_path:
                logger.info(f"Saved video chunk: {os.path.basename(previous_chunk_path)}")

        output_directory = target_dir or self.output_dir
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_video_path = os.path.join(output_directory, f"{self.FILENAME_PREFIX}{timestamp}{self.FILENAME_SUFFIX}")
        fourcc = cv2.VideoWriter_fourcc(*self.VIDEO_CODEC)
        self.writer = cv2.VideoWriter(self.current_video_path, fourcc, self.fps, frame_size)
        self.chunk_start_time = time.time()
        self.is_new_chunk = True
        logger.info(f"Started new video chunk: {os.path.basename(self.current_video_path)}")
        return previous_chunk_path

    def _write_frame_blocking(self, frame: np.ndarray, target_dir: str | None = None) -> str | None:
        """Writes a single frame, starting a new chunk if necessary.

        Args:
            frame: The video frame (as a NumPy array) to be written.
            target_dir: The target directory for new chunks, if one needs to be started.

        Returns:
            The path of a video chunk if it was just closed, otherwise None.
        """
        closed_chunk_path: str | None = None
        is_new_chunk_needed = self.writer is None or (time.time() - self.chunk_start_time) >= self.chunk_length_seconds

        if is_new_chunk_needed:
            height, width, _ = frame.shape
            closed_chunk_path = self._start_new_chunk_blocking(frame_size=(width, height), target_dir=target_dir)

        if self.writer:
            self.writer.write(frame)
            if self.is_new_chunk and target_dir is None and self.pre_fire_buffer is not None:
                self.pre_fire_buffer.append(self.current_video_path)
                self.is_new_chunk = False
        return closed_chunk_path

    def reset_after_fire(self) -> None:
        """Resets the state after handling a fire event."""
        self.signal_handler.reset_fire_event()
        self.signal_handler.reset_fire_extinguished_event()
        self.strategy.reset()

    def __call__(self, frame: np.ndarray) -> None:
        """Delegates frame handling to the current strategy object.

        Args:
            frame: The video frame to process.
        """
        if self.enabled:
            self.strategy.add_frame(frame)

    def _flush_buffer_to_disk_blocking(self, frames: deque[bytes], path: str) -> None:
        """Writes a deque of compressed frames to a single video file.

        Args:
            frames: A deque of JPEG-compressed video frames (as bytes) to write.
            path: The full path of the output video file.
        """
        if not frames:
            logger.warning("Memory buffer is empty, cannot flush to disk.")
            return

        # Decode the first frame to get the video dimensions
        first_frame_array = np.frombuffer(frames[0], dtype=np.uint8)
        first_frame = cv2.imdecode(first_frame_array, cv2.IMREAD_COLOR)
        if first_frame is None:
            logger.error("Could not decode first frame from memory buffer. Aborting flush.")
            return

        height, width, _ = first_frame.shape
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*self.VIDEO_CODEC)
        writer = cv2.VideoWriter(path, fourcc, self.fps, frame_size)
        if not writer.isOpened():
            logger.error(f"Failed to open video writer for path: {path}")
            return

        logger.info(f"Flushing {len(frames)} compressed frames from memory to {os.path.basename(path)}...")
        try:
            # Write the already decoded first frame
            writer.write(first_frame)
            # Decode and write the rest of the frames.
            for i, encoded_frame in enumerate(itertools.islice(frames, 1, None), start=1):
                frame_array = np.frombuffer(encoded_frame, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    writer.write(frame)
                else:
                    logger.warning(f"Could not decode frame index {i} from memory buffer. Skipping.")
        finally:
            writer.release()
            logger.info(f"Finished flushing buffer to {os.path.basename(path)}.")

    @staticmethod
    def _move_file_blocking(chunk_path: str, event_dir: str) -> None:
        """Moves a file to an event archive directory."""
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
            await self.strategy.handle_fire_event(event_dir)
            await self._save_post_fire_chunks_async(event_dir)
            logger.info("Fire event handled. Restoring normal chunk rotation policy.")
        finally:
            self.chunk_limit_action = original_cleanup_action

    async def _finalize_active_chunk_async(self, event_dir: str) -> None:
        """Continues recording the active chunk and archives it upon completion.

        Args:
            event_dir: The destination directory for the archive.
        """
        logger.info("Finalizing the video chunk active during fire detection...")
        loop = asyncio.get_running_loop()
        timeout_retries = 0
        while True:
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=self.POST_FIRE_FRAME_TIMEOUT)
                if frame is None:
                    logger.warning("Shutdown signaled while finalizing active chunk.")
                    return
                closed_chunk = await loop.run_in_executor(None, self._write_frame_blocking, frame, event_dir)
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

    async def _record_during_fire_loop(self, event_dir: str) -> None:
        """Records video chunks until the 'fire extinguished' signal is received.

        Args:
            event_dir: The directory where video chunks will be saved.
        """
        logger.info("Recording will continue for the duration of the fire.")
        loop = asyncio.get_running_loop()
        while not self.signal_handler.is_fire_extinguished():
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=self.FRAME_QUEUE_POLL_TIMEOUT)
                if frame is None:
                    logger.warning("Shutdown signaled during primary fire recording.")
                    return
                await loop.run_in_executor(None, self._write_frame_blocking, frame, event_dir)
            except TimeoutError:
                continue
        logger.info("Fire extinguished signal received. Saving final post-fire chunks.")

    async def _record_final_chunks_loop(self, event_dir: str) -> None:
        """Saves a configured number of frames, equivalent to a total duration.

        Args:
            event_dir: The directory where video chunks will be saved.
        """
        if self.chunks_to_keep_after_fire <= 0:
            return

        total_duration_seconds = self.chunks_to_keep_after_fire * self.chunk_length_seconds
        total_frames_to_record = int(total_duration_seconds * self.fps)

        logger.info(f"Now saving ~{total_duration_seconds} seconds of post-fire video ({total_frames_to_record} frames)...")
        loop = asyncio.get_running_loop()
        frames_recorded = 0
        while frames_recorded < total_frames_to_record:
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=self.POST_FIRE_FRAME_TIMEOUT)
                if frame is None:
                    logger.warning("Shutdown signaled during post-fire recording. Finalizing early.")
                    break
                await loop.run_in_executor(None, self._write_frame_blocking, frame, event_dir)
                frames_recorded += 1
            except TimeoutError:
                logger.warning(
                    f"Stream ended during post-fire recording. "
                    f"Saved {frames_recorded}/{total_frames_to_record} frames before timeout."
                )
                break

    async def _flush_remaining_frames(self) -> None:
        """Drains the frame queue and writes any remaining frames to the current writer."""
        if self.writer is None:
            return

        loop = asyncio.get_running_loop()
        if not self.frame_queue.empty():
            logger.info(f"Flushing {self.frame_queue.qsize()} remaining frames from queue to finalize video chunk...")
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                    if frame is not None:
                        # Writing is blocking, so run in executor. The target_dir doesn't
                        # matter here as we are not starting a new chunk.
                        await loop.run_in_executor(None, self.writer.write, frame)
                except asyncio.QueueEmpty:
                    break
            logger.info("Finished flushing remaining frames.")

    async def _save_post_fire_chunks_async(self, event_dir: str) -> None:
        """Saves additional chunks directly to the event directory after a fire."""
        try:
            if settings.video.record_during_fire:
                await self._record_during_fire_loop(event_dir)

            await self._record_final_chunks_loop(event_dir)
        finally:
            # This block is critical. It ensures that after all recording loops finish
            # (or are interrupted), any frames still buffered in the queue are written
            # out, and the video file is properly finalized and closed.
            await self._flush_remaining_frames()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._release_writer)
