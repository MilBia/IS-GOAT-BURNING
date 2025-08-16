import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import logging as log
import os
import shutil
import time

import cv2
from vidgear.gears.helper import logger_handler

from config import settings
from fire_detection.signal_handler import SignalHandler

logger = log.getLogger("AsyncVideoChunkSaver")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class AsyncVideoChunkSaver:
    """
    Manages saving video stream frames into timed chunks.
    """

    # --- Configuration ---
    enabled: bool = settings.video.save_video_chunks
    output_dir: str = settings.video.video_output_directory
    chunk_length_seconds: int = settings.video.video_chunk_length_seconds
    max_chunks: int = settings.video.max_video_chunks
    chunks_to_keep_after_fire: int = settings.video.chunks_to_keep_after_fire
    fps: float = 30.0
    FILENAME_PREFIX: str = "goat-cam_"
    FILENAME_SUFFIX: str = ".mp4"
    MAX_TIMEOUT_RETRIES: int = 3

    # --- Internal State ---
    frame_queue: asyncio.Queue = field(init=False, default_factory=asyncio.Queue)
    archive_queue: asyncio.Queue = field(init=False, default_factory=asyncio.Queue)
    writer: cv2.VideoWriter = field(init=False, default=None, repr=False)
    chunk_start_time: float = field(init=False, default=0.0)
    current_video_path: str = field(init=False, default=None)
    _task: asyncio.Task = field(init=False, default=None, repr=False)
    _archive_task: asyncio.Task = field(init=False, default=None, repr=False)
    __call__: Callable
    chunk_limit_action: Callable = field(init=False, default=None, repr=False)
    signal_handler: SignalHandler = field(init=False, default_factory=SignalHandler)
    pre_fire_buffer: deque = field(init=False)
    is_new_chunk: bool = field(init=False, default=False)

    def __post_init__(self):
        self.pre_fire_buffer = deque(maxlen=self.max_chunks)
        self.__call__ = self._noop
        self.chunk_limit_action = self._noop
        if not self.enabled:
            return

        self.__call__ = self._write_frame
        self.create_storage_directory()
        if self.max_chunks > 0:
            self.chunk_limit_action = self._enforce_chunk_limit_blocking

    def create_storage_directory(self) -> None:
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Video chunks will be saved to: {self.output_dir}")
        except OSError as e:
            logger.error(f"Could not create directory {self.output_dir}: {e}")
            self.__call__ = self._noop  # Disable if directory creation fails

    def start(self):
        """Starts the background writer task."""
        if not self.enabled:
            return
        # Create a background task that will consume frames from the queue
        self._task = asyncio.create_task(self._writer_task())
        self._archive_task = asyncio.create_task(self.archive_task())

    async def stop(self):
        """Signals the writer task to stop and waits for it to finish."""
        if not self.enabled or not self._task:
            return

        # Signal the writer to finish by putting a sentinel value (None) in the queue
        await self.frame_queue.put(None)
        await self.archive_queue.put(None)
        # Wait for the task to complete
        await asyncio.gather(
            asyncio.shield(self._task),
            asyncio.shield(self._archive_task),
            return_exceptions=True,
        )

    def _noop(self, *args, **kwargs) -> None:
        """A "no-operation" method that does nothing, used when saving is disabled."""
        pass

    def _enforce_chunk_limit_blocking(self):
        """
        Checks and removes the oldest chunk(s) if the max limit is reached.
        """
        try:
            files = sorted(
                f
                for f in os.listdir(self.output_dir)
                if f.startswith(self.FILENAME_PREFIX)
                and f.endswith(self.FILENAME_SUFFIX)
                and os.path.isfile(os.path.join(self.output_dir, f))
            )

            # Calculate the number of files to delete.
            num_to_delete = len(files) - self.max_chunks + 1

            if num_to_delete > 0:
                # Iterate over a slice of the oldest files.
                for oldest_file in files[:num_to_delete]:
                    file_path_to_delete = os.path.join(self.output_dir, oldest_file)
                    # Remove the oldest file
                    try:
                        os.remove(file_path_to_delete)
                        logger.debug(f"Removed oldest chunk: {oldest_file}")
                    except OSError as e:
                        # Log errors on a per-file basis.
                        logger.error(f"Could not remove file {file_path_to_delete}: {e}")

        except OSError as e:
            # Catches critical errors from os.listdir().
            logger.error(f"Error listing files in {self.output_dir} for cleanup: {e}")

    def _start_new_chunk_blocking(self, frame_size: tuple):
        """
        Synchronous method to set up the cv2.VideoWriter.

        Args:
            frame_size (tuple): A tuple of (width, height) for the video frame.
        """
        # Enforce the chunk limit before creating a new file
        self.chunk_limit_action()

        # Release the previous writer if it exists
        previous_chunk_path = self.current_video_path
        if self.writer is not None:
            self.writer.release()
            logger.info(f"Saved video chunk: {os.path.basename(previous_chunk_path)}")

        # Generate a new timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_video_path = os.path.join(self.output_dir, f"{self.FILENAME_PREFIX}{timestamp}{self.FILENAME_SUFFIX}")

        # Define the codec and create the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.current_video_path, fourcc, self.fps, frame_size)

        # Reset the chunk timer
        self.chunk_start_time = time.time()
        self.is_new_chunk = True
        logger.info(f"Started new video chunk: {os.path.basename(self.current_video_path)}")
        return previous_chunk_path

    def _write_frame_blocking(self, frame) -> str | None:
        """The actual blocking I/O call. This runs in the thread pool."""
        closed_chunk_path = None
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

    def _write_frame(self, frame) -> None:
        try:
            # Put frame in the queue without blocking the main loop
            self.frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            # This can happen if the writer task can't keep up.
            # You might want to log this or drop the frame.
            logger.warning("Frame queue is full, dropping a frame.")

    def reset_after_fire(self):
        self.pre_fire_buffer.clear()
        self.signal_handler.reset_fire_event()
        logger.debug("FIRE EVENT HANDLING COMPLETE. Resuming normal operations.")

    async def _writer_task(self):
        """The main consumer task that runs in the background."""
        loop = asyncio.get_running_loop()
        while self.signal_handler.KEEP_PROCESSING:
            # Check for the fire event signal first.
            if self.signal_handler.is_fire_detected():
                # Disable normal chunk cleanup during event handling
                original_cleanup_action = self.chunk_limit_action
                self.chunk_limit_action = self._noop
                try:
                    await self._handle_fire_event_async()
                finally:
                    # Re-enable normal chunk cleanup after event is handled
                    self.reset_after_fire()
                    self.chunk_limit_action = original_cleanup_action
                    logger.info("Fire event handled. Restoring normal chunk rotation policy.")

            try:
                # Wait for a frame with a timeout to remain responsive to signals.
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=0.1)
                if frame is None:  # Graceful stop via queue
                    break
                # Run the blocking write operation in a separate thread
                await loop.run_in_executor(None, self._write_frame_blocking, frame)

            except asyncio.TimeoutError:
                continue  # No frame, just loop and check signals again.
            except asyncio.CancelledError:
                logger.info("Writer task was cancelled.")
                break  # Exit if task is cancelled
            except Exception as e:
                logger.error(f"Error writing frame: {e}", exc_info=True)
                break  # Exit the writer task on error to prevent repeated failures

        # Final cleanup when the loop is broken
        if self.writer is not None:
            self.writer.release()
            logger.info(f"Saved final video chunk on exit: {self.current_video_path}")

    def __call__(self, frame):
        """
        Placeholder for the __call__ method.
        This will be overwritten in __init__ for each instance.
        Calling it on a non-initialized instance will raise an error.
        """
        raise NotImplementedError("This class must be initialized before it can be called.")

    @staticmethod
    def _move_file_blocking(chunk_path: str, event_dir: str):
        """Moves a single file and handles errors. Designed for the archiver."""
        try:
            if os.path.exists(chunk_path):
                shutil.move(chunk_path, event_dir)
                logger.debug(f"Archived {os.path.basename(chunk_path)}.")
            else:
                logger.warning(f"Chunk {chunk_path} not found for archiving.")
        except (OSError, shutil.Error) as e:
            logger.error(f"Could not move chunk {chunk_path}: {e}")

    async def archive_task(self):
        """A dedicated background task that consumes from the archive_queue and moves files."""
        loop = asyncio.get_running_loop()
        while True:
            item = await self.archive_queue.get()

            if item is None:  # Graceful shutdown signal
                break

            # Otherwise, it's a tuple of (path, destination)
            chunk_path, event_dir = item
            await loop.run_in_executor(None, self._move_file_blocking, chunk_path, event_dir)

    async def _handle_fire_event_async(self):
        """Orchestrates concurrent archiving and continued recording."""
        logger.warning("FIRE EVENT TRIGGERED! Archiving will occur in the background.")

        # 1. Create the event directory immediately. This is fast.
        event_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        event_dir = os.path.join(self.output_dir, f"event_{event_timestamp}")
        try:
            os.makedirs(event_dir, exist_ok=True)
            logger.info(f"Created event archive directory: {event_dir}")
        except OSError as e:
            logger.error(f"Could not create event directory {event_dir}: {e}. Aborting archive.")
            self.signal_handler.reset_fire_event()
            return

        # 2. Identify "cold" vs "hot" files and start moving cold files immediately.
        cold_chunks = list(self.pre_fire_buffer)[:-1]
        logger.info(f"Queueing {len(cold_chunks)} pre-fire chunks for archiving.")
        for path in cold_chunks:
            try:
                self.archive_queue.put_nowait((path, event_dir))
            except asyncio.QueueFull:
                logger.error(f"Archive queue is full. Failed to queue chunk for archiving {path}.")

        # 3. Call the specialized handlers for live chunks seving
        await self._finalize_active_chunk_async(event_dir)
        await self._save_post_fire_chunks_async(event_dir)

    async def _finalize_active_chunk_async(self, event_dir: str):
        """
        Phase 1 of fire event handling: continues recording until the currently
        active video chunk is finalized and queued for archiving.
        """
        logger.info("Finalizing the video chunk active during fire detection...")
        loop = asyncio.get_running_loop()
        active_chunk_saved = False
        timeout_retries = 0

        while not active_chunk_saved:
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)

                timeout_retries = 0

                if frame is None or not self.signal_handler.KEEP_PROCESSING:
                    logger.warning("Shutdown signaled while finalizing active chunk.")
                    break  # Exit this loop

                closed_chunk = await loop.run_in_executor(None, self._write_frame_blocking, frame)

                if closed_chunk:
                    # The active chunk is now closed and saved. Queue it for archiving.
                    logger.info("Active chunk finalized and queued for archive.")
                    try:
                        self.archive_queue.put_nowait((closed_chunk, event_dir))
                    except asyncio.QueueFull:
                        logger.error(f"Archive queue is full. Failed to queue chunk for archiving {closed_chunk}.")
                    active_chunk_saved = True  # Signal to exit this loop

            except asyncio.TimeoutError:
                timeout_retries += 1
                logger.warning(
                    f"Timeout waiting for frame while finalizing active chunk. "
                    f"Retry {timeout_retries}/{self.MAX_TIMEOUT_RETRIES}..."
                )
                if timeout_retries >= self.MAX_TIMEOUT_RETRIES:
                    logger.error("Timeout waiting for frame while finalizing active chunk. Aborting event.")
                    break  # Exit this loop

    async def _save_post_fire_chunks_async(self, event_dir: str):
        """
        Phase 2 of fire event handling: saves a configured number of additional
        chunks after the initial event.
        """
        if self.chunks_to_keep_after_fire <= 0:
            return

        logger.info(f"Now saving {self.chunks_to_keep_after_fire} post-fire chunks...")
        loop = asyncio.get_running_loop()
        chunks_saved_count = 0
        timeout_retries = 0

        while chunks_saved_count < self.chunks_to_keep_after_fire:
            try:
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)

                timeout_retries = 0

                if frame is None or not self.signal_handler.KEEP_PROCESSING:
                    logger.warning("Shutdown signaled during post-fire recording.")
                    break

                # Write the frame and get the path of the chunk that was just closed.
                closed_chunk = await loop.run_in_executor(None, self._write_frame_blocking, frame)

                # The newly closed chunk is now "cold" and can be archived.
                if closed_chunk:
                    chunks_saved_count += 1
                    try:
                        self.archive_queue.put_nowait((closed_chunk, event_dir))
                    except asyncio.QueueFull:
                        logger.error(f"Archive queue is full. Failed to queue chunk for archiving {closed_chunk}.")

            except asyncio.TimeoutError:
                timeout_retries += 1
                logger.warning(
                    f"Timeout waiting for frame while finalizing active chunk. "
                    f"Retry {timeout_retries}/{self.MAX_TIMEOUT_RETRIES}..."
                )
                if timeout_retries >= self.MAX_TIMEOUT_RETRIES:
                    logger.error("Timeout waiting for frame during post-fire recording.")
                    break
