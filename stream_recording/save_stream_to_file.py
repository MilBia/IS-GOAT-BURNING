import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import logging as log
import os
import time

import cv2
from vidgear.gears.helper import logger_handler

from setting import MAX_VIDEO_CHUNKS
from setting import SAVE_VIDEO_CHUNKS
from setting import VIDEO_CHUNK_LENGTH_SECONDS
from setting import VIDEO_OUTPUT_DIRECTORY

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
    enabled: bool = SAVE_VIDEO_CHUNKS
    output_dir: str = VIDEO_OUTPUT_DIRECTORY
    chunk_length_seconds: int = VIDEO_CHUNK_LENGTH_SECONDS
    max_chunks: int = MAX_VIDEO_CHUNKS
    fps: float = 30.0

    # --- Internal State ---
    frame_queue: asyncio.Queue = field(init=False, default_factory=asyncio.Queue)
    writer: cv2.VideoWriter = field(init=False, default=None, repr=False)
    chunk_start_time: float = field(init=False, default=0.0)
    current_video_path: str = field(init=False, default=None)
    _task: asyncio.Task = field(init=False, default=None, repr=False)
    __call__: Callable
    chunk_limit_action: Callable = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.enabled:
            self.__call__ = self._write_frame
            self.create_storage_directory()
            print(f"{self.max_chunks <= 0=}")
            if self.max_chunks > 0:
                print("A")
                self.chunk_limit_action = self._enforce_chunk_limit_blocking
            else:
                print("B")
                self.chunk_limit_action = self._noop
        else:
            self.__call__ = self._noop

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

    async def stop(self):
        """Signals the writer task to stop and waits for it to finish."""
        if not self.enabled or not self._task:
            return

        # Signal the writer to finish by putting a sentinel value (None) in the queue
        await self.frame_queue.put(None)
        # Wait for the task to complete
        await self._task

    def _enforce_chunk_limit_blocking(self):
        """Checks and removes the oldest chunk(s) if the max limit is reached."""
        try:
            # Get all video chunks matching the naming convention
            files = [f for f in os.listdir(self.output_dir) if f.startswith("goat-cam_") and f.endswith(".mp4")]

            # Sort files alphabetically to find the oldest.
            # This works because of the YYYY-MM-DD_HH-MM-SS timestamp format.
            files.sort()

            # Remove the oldest files until we are under the limit.
            while len(files) >= self.max_chunks:
                oldest_file = files.pop(0)
                file_path_to_delete = os.path.join(self.output_dir, oldest_file)
                os.remove(file_path_to_delete)
                logger.info(f"Removed oldest chunk to maintain limit: {oldest_file}")

        except OSError as e:
            logger.error(f"Error enforcing chunk limit in {self.output_dir}: {e}")

    def _start_new_chunk_blocking(self, frame_size: tuple):
        """
        Synchronous method to set up the cv2.VideoWriter.

        Args:
            frame_size (tuple): A tuple of (width, height) for the video frame.
        """
        # Enforce the chunk limit before creating a new file
        self.chunk_limit_action()

        # Release the previous writer if it exists
        if self.writer is not None:
            self.writer.release()
            logger.info(f"Saved video chunk: {self.current_video_path}")

        # Generate a new timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_video_path = os.path.join(self.output_dir, f"goat-cam_{timestamp}.mp4")

        # Define the codec and create the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.current_video_path, fourcc, self.fps, frame_size)

        # Reset the chunk timer
        self.chunk_start_time = time.time()
        logger.info(f"Started new video chunk: {self.current_video_path}")

    def _write_frame_blocking(self, frame) -> None:
        """The actual blocking I/O call. This runs in the thread pool."""
        is_new_chunk_needed = self.writer is None or (time.time() - self.chunk_start_time) >= self.chunk_length_seconds

        if is_new_chunk_needed:
            height, width, _ = frame.shape
            self._start_new_chunk_blocking(frame_size=(width, height))

        if self.writer:
            self.writer.write(frame)

    def _write_frame(self, frame) -> None:
        try:
            # Put frame in the queue without blocking the main loop
            self.frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            # This can happen if the writer task can't keep up.
            # You might want to log this or drop the frame.
            logger.warning("Frame queue is full, dropping a frame.")

    def _noop(self, *args, **kwargs) -> None:
        """A "no-operation" method that does nothing, used when saving is disabled."""
        pass

    async def _writer_task(self):
        """The main consumer task that runs in the background."""
        loop = asyncio.get_running_loop()
        while True:
            # Wait for a frame from the queue
            frame = await self.frame_queue.get()

            # The sentinel value `None` signals the task to exit
            if frame is None:
                break

            try:
                # Run the blocking write operation in a separate thread
                await loop.run_in_executor(None, self._write_frame_blocking, frame)
            except asyncio.CancelledError:
                logger.info("Writer task was cancelled.")
                break  # Exit if task is cancelled
            except Exception as e:
                logger.error(f"Error writing frame: {e}")
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
