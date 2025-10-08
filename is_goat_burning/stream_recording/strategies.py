"""Defines the buffering strategies for the AsyncVideoChunkSaver.

This module implements the Strategy design pattern for video buffering. It provides
an abstract base class, `BufferStrategy`, and concrete implementations for
different buffering methods, such as writing to disk continuously
(`DiskBufferStrategy`) or holding frames in memory until an event
(`MemoryBufferStrategy`).
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import asyncio
from collections import deque
from datetime import datetime
import os
import shutil
from typing import TYPE_CHECKING

import cv2
import numpy as np

from is_goat_burning.config import settings
from is_goat_burning.logger import get_logger

if TYPE_CHECKING:
    from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver


logger = get_logger("BufferStrategy")


class BufferStrategy(ABC):
    """Abstract base class defining the interface for a video buffer strategy."""

    def __init__(self, context: AsyncVideoChunkSaver) -> None:
        """Initializes the strategy with a reference to its context.

        Args:
            context: The AsyncVideoChunkSaver instance that uses this strategy.
        """
        self.context = context
        self._main_task: asyncio.Task[None] | None = None

    @abstractmethod
    def add_frame(self, frame: np.ndarray) -> None:
        """Adds a frame to the buffer or processing queue.

        Args:
            frame: The video frame to be processed.
        """
        ...

    @abstractmethod
    async def _run_main_loop(self) -> None:
        """The core logic loop for the strategy (e.g., writing frames or waiting for events)."""
        ...

    @abstractmethod
    async def handle_fire_event(self, event_dir: str) -> None:
        """Orchestrates the archiving of pre-fire and post-fire chunks.

        Args:
            event_dir: The directory where event-related videos should be saved.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Resets the strategy's internal state after a fire event is handled."""
        ...

    async def start(self) -> None:
        """Starts the strategy's main processing loop as a background task."""
        if self._main_task is None or self._main_task.done():
            self._main_task = asyncio.create_task(self._run_main_loop())

    async def stop(self) -> None:
        """Stops the strategy's main processing loop."""
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                logger.debug("Strategy main task successfully cancelled.")


class DiskBufferStrategy(BufferStrategy):
    """A strategy that continuously writes incoming frames to disk chunks."""

    def add_frame(self, frame: np.ndarray) -> None:
        """Puts a frame onto the asynchronous queue to be written to disk.

        Args:
            frame: The video frame to be queued.
        """
        try:
            self.context.frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Frame queue is full, dropping a frame.")

    def reset(self) -> None:
        """Resets the pre-fire buffer and safely drains the frame queue."""
        if self.context.pre_fire_buffer is not None:
            self.context.pre_fire_buffer.clear()
        self.context._drain_frame_queue()

    async def handle_fire_event(self, event_dir: str) -> None:
        """Handles the fire event for 'disk' buffering mode.

        This method archives the already saved chunks that were captured before
        the fire started and finalizes the chunk that was active during the fire
        detection.

        Args:
            event_dir: The destination directory for the archived videos.
        """
        # Archive all chunks currently in the buffer except the active one.
        if self.context.pre_fire_buffer is not None:
            cold_chunks = list(self.context.pre_fire_buffer)[:-1]
            logger.info(f"Queueing {len(cold_chunks)} pre-fire chunks for archiving.")
            for path in cold_chunks:
                try:
                    self.context.archive_queue.put_nowait((path, event_dir))
                except asyncio.QueueFull:
                    logger.error(f"Archive queue is full. Failed to queue chunk for archiving {path}.")

        await self.context._finalize_active_chunk_async(event_dir)

    async def _handle_fire_signal(self) -> None:
        """Checks for and handles the fire detection signal."""
        if self.context.signal_handler.is_fire_detected() and not self.context._fire_handling_lock.locked():
            async with self.context._fire_handling_lock:
                if self.context.signal_handler.is_fire_detected():
                    try:
                        await self.context._handle_fire_event_async()
                    except asyncio.CancelledError:
                        raise
                    except (OSError, shutil.Error, cv2.error):
                        logger.exception("Unexpected but recoverable error during disk strategy fire event handling.")
                    finally:
                        self.context.reset_after_fire()

    async def _run_main_loop(self) -> None:
        """The main consumer task that writes frames from the queue to disk."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                await self._handle_fire_signal()
                try:
                    frame = await asyncio.wait_for(
                        self.context.frame_queue.get(), timeout=self.context.FRAME_QUEUE_POLL_TIMEOUT
                    )
                    await loop.run_in_executor(None, self.context._write_frame_blocking, frame)
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    logger.info("Disk strategy writer task cancelled during frame wait.")
                    raise
                except (OSError, cv2.error):
                    logger.exception("Error in disk strategy writer task, but will continue running.")
        except asyncio.CancelledError:
            logger.info("Disk strategy main loop was cancelled.")
            raise


class MemoryBufferStrategy(BufferStrategy):
    """A strategy that holds recent frames in memory and writes them to disk only on a fire event."""

    def __init__(self, context: AsyncVideoChunkSaver) -> None:
        """Initializes the memory buffer strategy.

        Args:
            context: The AsyncVideoChunkSaver instance that uses this strategy.
        """
        super().__init__(context)
        buffer_size = int(settings.video.memory_buffer_seconds * self.context.fps)
        self.memory_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._is_post_fire_recording: bool = False
        logger.info(f"Memory buffer initialized with a capacity for {buffer_size} frames.")

    def reset(self) -> None:
        """Resets the memory buffer, drains the frame queue, and restores initial behavior."""
        self.memory_buffer.clear()
        self._is_post_fire_recording = False
        self.context._drain_frame_queue()

    def add_frame(self, frame: np.ndarray) -> None:
        """Adds a frame to memory or the disk queue based on the current recording state."""
        if self._is_post_fire_recording:
            self._queue_frame(frame)
        else:
            self._add_frame_to_memory_buffer(frame)

    def _add_frame_to_memory_buffer(self, frame: np.ndarray) -> None:
        """Adds a frame to the in-memory buffer deque."""
        self.memory_buffer.append(frame.copy())

    def _queue_frame(self, frame: np.ndarray) -> None:
        """Queues a frame for disk writing, used during post-fire recording."""
        try:
            self.context.frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Frame queue is full, dropping a frame.")

    async def _flush_and_archive_memory_buffer(self, event_dir: str) -> None:
        """Flushes the in-memory frame buffer to a file in the event directory."""
        frames_to_flush = self.memory_buffer.copy()
        self.memory_buffer.clear()
        logger.info(f"Captured {len(frames_to_flush)} pre-fire frames from memory.")

        if not frames_to_flush:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = self.context.current_video_path = os.path.join(
            event_dir, f"{self.context.FILENAME_PREFIX}pre-fire-buffer_{timestamp}{self.context.FILENAME_SUFFIX}"
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.context._flush_buffer_to_disk_blocking, frames_to_flush, file_path)

    async def handle_fire_event(self, event_dir: str) -> None:
        """Handles the fire event for 'memory' buffering mode.

        This method flushes the in-memory buffer to a pre-fire video file and
        then switches the behavior to start queueing frames for the post-fire
        recording phase.

        Args:
            event_dir: The destination directory for the archived videos.
        """
        await self._flush_and_archive_memory_buffer(event_dir)
        logger.info("Memory buffer flushed. Switching to queueing mode for post-fire recording.")
        self._is_post_fire_recording = True

    async def _run_main_loop(self) -> None:
        """The main event loop for 'memory' buffer mode.

        This loop waits efficiently for the fire signal and then triggers the
        context's main event handling logic when the signal is set.
        """
        try:
            while True:
                await self.context.signal_handler.fire_detected_event.wait()

                async with self.context._fire_handling_lock:
                    if not self.context.signal_handler.is_fire_detected():
                        continue
                    try:
                        await self.context._handle_fire_event_async()
                    except asyncio.CancelledError:
                        raise
                    except (OSError, shutil.Error, cv2.error):
                        logger.exception("Unexpected but recoverable error during memory strategy fire event handling.")
                    finally:
                        self.context.reset_after_fire()
        except asyncio.CancelledError:
            logger.info("Memory strategy main loop was cancelled.")
            raise
