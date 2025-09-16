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
import logging as log
import os
import shutil
import time
from typing import Any
from typing import ClassVar

import cv2
import numpy as np
from vidgear.gears.helper import logger_handler

from is_goat_burning.fire_detection.signal_handler import SignalHandler

logger = log.getLogger("AsyncVideoChunkSaver")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class AsyncVideoChunkSaver:
    """Manages saving a video stream into timed, rotating file chunks.

    This class provides an asynchronous interface to save video frames. It uses
    a background asyncio task to consume frames from a queue and write them to
    disk, preventing I/O from blocking the main application loop. It also
    integrates with the `SignalHandler` to perform special archiving logic when
    a fire is detected.

    Attributes:
        enabled (bool): If False, all operations are no-ops.
        output_dir (str): The directory where video chunks are saved.
        chunk_length_seconds (int): The duration of each video chunk file.
        max_chunks (int): The maximum number of non-archived chunks to keep on
            disk. Older chunks are deleted to enforce this limit.
        chunks_to_keep_after_fire (int): The number of additional chunks to
            record and archive after a fire is first detected.
        fps (float): The frames per second of the video stream.
        frame_queue (asyncio.Queue): A queue for incoming frames to be written.
        archive_queue (asyncio.Queue): A queue for video chunks to be archived.
        writer (cv2.VideoWriter): The OpenCV video writer instance.
        pre_fire_buffer (deque): A deque that stores the paths of the most
            recent video chunks, used for pre-event archiving.
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
    MAX_TIMEOUT_RETRIES: ClassVar[int] = 3

    # --- Internal State ---
    frame_queue: asyncio.Queue[np.ndarray | None] = field(init=False, default_factory=asyncio.Queue)
    archive_queue: asyncio.Queue[tuple[str, str] | None] = field(init=False, default_factory=asyncio.Queue)
    writer: cv2.VideoWriter | None = field(init=False, default=None, repr=False)
    chunk_start_time: float = field(init=False, default=0.0)
    current_video_path: str | None = field(init=False, default=None)
    _task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _archive_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    __call__: Callable[[np.ndarray], None]
    chunk_limit_action: Callable[[], None] = field(init=False, repr=False)
    makedirs_callable: Callable[..., None] = field(init=False, repr=False)
    signal_handler: SignalHandler = field(init=False, default_factory=SignalHandler)
    pre_fire_buffer: deque[str] = field(init=False)
    is_new_chunk: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Initializes state and methods based on the `enabled` flag."""
        self.pre_fire_buffer = deque(maxlen=self.max_chunks)
        self.__call__ = self._noop
        self.chunk_limit_action = self._noop
        self.makedirs_callable = partial(os.makedirs, exist_ok=True)
        if not self.enabled:
            return

        self.__call__ = self._queue_frame
        self.create_storage_directory()
        if self.max_chunks > 0:
            self.chunk_limit_action = self._enforce_chunk_limit_blocking

    def create_storage_directory(self) -> None:
        """Creates the output directory if it doesn't exist."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Video chunks will be saved to: {self.output_dir}")
        except OSError as e:
            logger.error(f"Could not create directory {self.output_dir}: {e}")
            self.__call__ = self._noop  # Disable if directory creation fails

    def start(self) -> None:
        """Starts the background writer and archiver tasks."""
        if not self.enabled:
            return
        self._task = asyncio.create_task(self._writer_task())
        self._archive_task = asyncio.create_task(self.archive_task())

    async def stop(self) -> None:
        """Signals the background tasks to stop and waits for them to finish."""
        if not self.enabled or not self._task:
            return

        # Signal the tasks to finish by putting a sentinel value (None)
        await self.frame_queue.put(None)
        await self.archive_queue.put(None)
        await asyncio.gather(
            asyncio.shield(self._task),
            asyncio.shield(self._archive_task),
            return_exceptions=True,
        )

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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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
        """Puts a frame onto the asynchronous queue to be written.

        This is the main entry point for the class when it's active.

        Args:
            frame: The video frame to be saved.
        """
        try:
            self.frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Frame queue is full, dropping a frame.")

    def reset_after_fire(self) -> None:
        """Resets the state after handling a fire event."""
        self.pre_fire_buffer.clear()
        self.signal_handler.reset_fire_event()
        logger.debug("FIRE EVENT HANDLING COMPLETE. Resuming normal operations.")

    async def _writer_task(self) -> None:
        """The main consumer task that writes frames from the queue to disk."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                if self.signal_handler.is_fire_detected():
                    original_cleanup_action = self.chunk_limit_action
                    self.chunk_limit_action = self._noop
                    try:
                        await self._handle_fire_event_async()
                    finally:
                        self.reset_after_fire()
                        self.chunk_limit_action = original_cleanup_action
                        logger.info("Fire event handled. Restoring normal chunk rotation policy.")
                try:
                    frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1)
                    if frame is None:  # Graceful stop sentinel
                        break
                    await loop.run_in_executor(None, self._write_frame_blocking, frame)
                except TimeoutError:
                    continue  # No frame, loop to check for fire signal again.
        except asyncio.CancelledError:
            logger.info("Writer task was cancelled.")
            raise
        finally:
            if self.writer is not None:
                self.writer.release()
                if self.current_video_path:
                    logger.info(f"Saved final video chunk on exit: {self.current_video_path}")

    def __call__(self, frame: np.ndarray) -> None:
        """Allows the class instance to be called like a function.

        This method is dynamically assigned to either `_queue_frame` or `_noop`
        during initialization based on the `enabled` flag.

        Args:
            frame: The video frame to process.
        """
        raise NotImplementedError("This class must be initialized before it can be called.")

    @staticmethod
    def _move_file_blocking(chunk_path: str, event_dir: str) -> None:
        """Moves a file from the main directory to an event archive directory.

        This is a synchronous method designed to be run in an executor by the
        archiver task.

        Args:
            chunk_path: The full path to the video chunk to move.
            event_dir: The destination directory for the archive.
        """
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
            if item is None:  # Graceful shutdown sentinel
                break
            chunk_path, event_dir = item
            await loop.run_in_executor(None, self._move_file_blocking, chunk_path, event_dir)

    async def _handle_fire_event_async(self) -> None:
        """Orchestrates the archiving of pre-fire and post-fire chunks."""
        logger.warning("FIRE EVENT TRIGGERED! Archiving will occur in the background.")
        event_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        event_dir = os.path.join(self.output_dir, f"event_{event_timestamp}")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.makedirs_callable, event_dir)
            logger.info(f"Created event archive directory: {event_dir}")
        except OSError as e:
            logger.error(f"Could not create event directory {event_dir}: {e}. Aborting archive.")
            self.signal_handler.reset_fire_event()
            return

        cold_chunks = list(self.pre_fire_buffer)[:-1]
        logger.info(f"Queueing {len(cold_chunks)} pre-fire chunks for archiving.")
        for path in cold_chunks:
            try:
                self.archive_queue.put_nowait((path, event_dir))
            except asyncio.QueueFull:
                logger.error(f"Archive queue is full. Failed to queue chunk for archiving {path}.")

        await self._finalize_active_chunk_async(event_dir)
        await self._save_post_fire_chunks_async(event_dir)

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
