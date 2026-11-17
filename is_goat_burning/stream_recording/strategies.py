"""Defines the buffering strategies for the AsyncVideoChunkSaver.

This module implements the Strategy design pattern for video buffering. It provides
an abstract base class, `BufferStrategy`, and concrete implementations for
different buffering methods, such as writing to disk continuously
(`DiskBufferStrategy`) or holding frames in memory until an event
(`MemoryBufferStrategy`).
"""

from abc import ABC
from abc import abstractmethod
import asyncio
from collections import deque
import contextlib
from datetime import datetime
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import TYPE_CHECKING
from typing import ClassVar

import cv2
import numpy as np

from is_goat_burning.config import settings
from is_goat_burning.logger import get_logger

if TYPE_CHECKING:
    from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver


logger = get_logger("BufferStrategy")


class BufferStrategy(ABC):
    """Abstract base class defining the interface for a video buffer strategy."""

    #: True when the strategy records the post-fire video itself (the saver then skips its frame-queue based recording).
    records_post_fire: ClassVar[bool] = False

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

    async def stop(self) -> None:
        """Stops the strategy's main processing loop gracefully, ensuring all frames are written."""
        if self._main_task and not self._main_task.done():
            # Use a sentinel value to ensure the queue is drained before stopping.
            await self.context.frame_queue.put(None)
            # Wait for the task to finish processing the queue.
            await self._main_task

    def add_frame(self, frame: np.ndarray) -> None:
        """Puts a raw frame onto the asynchronous queue to be written to disk.

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
                        logger.debug("Disk strategy fire event handling completed successfully.")
                    except asyncio.CancelledError:
                        raise
                    except (OSError, shutil.Error, cv2.error):  # fmt: skip
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
                    if frame is None:  # Sentinel for graceful stop
                        break

                    frames = [frame]
                    # Try to fetch more frames to process in a batch
                    while not self.context.frame_queue.empty() and len(frames) < self.context.FRAME_WRITE_BATCH_SIZE:
                        try:
                            next_frame = self.context.frame_queue.get_nowait()
                            if next_frame is None:
                                # Process collected frames before stopping
                                await loop.run_in_executor(None, self.context._write_frames_blocking, frames)
                                return
                            frames.append(next_frame)
                        except asyncio.QueueEmpty:
                            break

                    await loop.run_in_executor(None, self.context._write_frames_blocking, frames)
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    logger.info("Disk strategy writer task cancelled during frame wait.")
                    raise
                except (OSError, cv2.error):  # fmt: skip
                    logger.exception("Error in disk strategy writer task, but will continue running.")
        except asyncio.CancelledError:
            logger.info("Disk strategy main loop was cancelled.")
            raise
        finally:
            self.context._release_writer()


class MemoryBufferStrategy(BufferStrategy):
    """A strategy that holds recent frames in memory and writes them to disk only on a fire event."""

    JPEG_ENCODING_FORMAT: ClassVar[str] = ".jpg"

    def __init__(self, context: AsyncVideoChunkSaver) -> None:
        """Initializes the memory buffer strategy.

        Args:
            context: The AsyncVideoChunkSaver instance that uses this strategy.
        """
        super().__init__(context)
        buffer_size = int(settings.video.memory_buffer_seconds * self.context.fps)
        self.memory_buffer: deque[bytes] = deque(maxlen=buffer_size)
        self._is_post_fire_recording: bool = False
        logger.info(f"Memory buffer initialized with a capacity for {buffer_size} compressed frames.")

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
        """Compresses a frame to JPEG and adds it to the in-memory buffer deque."""
        success, encoded_image = cv2.imencode(self.JPEG_ENCODING_FORMAT, frame)
        if success:
            self.memory_buffer.append(encoded_image.tobytes())

    def _queue_frame(self, frame: np.ndarray) -> None:
        """Queues a compressed frame for disk writing, used during post-fire recording."""
        success, encoded_image = cv2.imencode(self.JPEG_ENCODING_FORMAT, frame)
        if not success:
            logger.warning("Failed to encode frame for queueing.")
            return

        try:
            self.context.frame_queue.put_nowait(encoded_image.tobytes())
        except asyncio.QueueFull:
            logger.warning("Frame queue is full, dropping a frame.")

    async def _flush_and_archive_memory_buffer(self, event_dir: str) -> None:
        """Flushes the in-memory buffer of compressed frames to a file in the event directory."""
        frames_to_flush = self.memory_buffer
        # Create a new deque for incoming frames, effectively clearing the buffer
        # without copying the data.
        buffer_size = self.memory_buffer.maxlen
        self.memory_buffer = deque(maxlen=buffer_size)
        logger.info(f"Captured {len(frames_to_flush)} compressed pre-fire frames from memory.")

        if not frames_to_flush:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = self.context.current_video_path = os.path.join(
            event_dir, f"{self.context.FILENAME_PREFIX}pre-fire-buffer_{timestamp}{self.context.FILENAME_SUFFIX}"
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.context._flush_buffer_to_disk_blocking, frames_to_flush, file_path)
        # The pre-fire buffer has been written; publish it for event-driven actions.
        self.context._publish_chunk_path(file_path)

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
                        logger.debug("Memory strategy fire event handling completed successfully.")
                    except asyncio.CancelledError:
                        raise
                    except (OSError, shutil.Error, cv2.error):  # fmt: skip
                        logger.exception("Unexpected but recoverable error during memory strategy fire event handling.")
                    finally:
                        self.context.reset_after_fire()
        except asyncio.CancelledError:
            logger.info("Memory strategy main loop was cancelled.")
            raise
        finally:
            self.context._release_writer()


class SegmentsBufferStrategy(BufferStrategy):
    """Archives the source stream without decoding it, using a ring of segments in RAM (tmpfs).

    One supervised ``ffmpeg -c copy`` process writes the original stream into short ``.ts`` segments that wrap around
    in ``settings.video.segments_directory``. Frames never reach `add_frame` (the streamer decodes only the frames it
    analyses), so archiving costs almost no CPU and a small, bounded amount of RAM. On a fire event the newest
    segments are concatenated with ``-c copy`` into the pre-fire clip, and the segments written afterwards are
    collected into post-fire chunks until the fire is out (when ``record_during_fire``) plus the configured tail.
    """

    records_post_fire: ClassVar[bool] = True
    SEGMENT_GLOB: ClassVar[str] = "seg*.ts"
    RING_EXTRA_SEGMENTS: ClassVar[int] = 4
    RESTART_DELAY_MIN: ClassVar[float] = 1.0
    RESTART_DELAY_MAX: ClassVar[float] = 30.0
    COLLECT_INTERVAL_SECONDS: ClassVar[float] = 5.0
    STABLE_RUN_SECONDS: ClassVar[float] = 30.0
    MIN_FREE_BYTES: ClassVar[int] = 128 * 1024 * 1024

    def __init__(self, context: AsyncVideoChunkSaver) -> None:
        """Initializes the segment ring parameters.

        Args:
            context: The AsyncVideoChunkSaver instance that uses this strategy.
        """
        super().__init__(context)
        self.ring_dir = Path(settings.video.segments_directory)
        self.segment_seconds = max(1, settings.video.segment_seconds)
        self.buffer_seconds = settings.video.memory_buffer_seconds
        self.ring_size = math.ceil(self.buffer_seconds / self.segment_seconds) + self.RING_EXTRA_SEGMENTS
        self._ffmpeg: asyncio.subprocess.Process | None = None
        self._supervisor: asyncio.Task[None] | None = None
        logger.info(f"Segments ring: {self.ring_size} slots of >= {self.segment_seconds}s in {self.ring_dir}.")

    def add_frame(self, frame: np.ndarray) -> None:
        """Does nothing: archiving is done by ffmpeg from the source, not from decoded frames."""

    def reset(self) -> None:
        """Nothing to reset: the ring keeps rolling across fire events."""

    def ffmpeg_command(self) -> list[str]:
        """Builds the ffmpeg command that stream-copies the source into the segment ring."""
        assert self.context.source_url is not None
        return [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "5",
            "-rw_timeout",
            "15000000",
            "-i",
            self.context.source_url,
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_seconds),
            "-segment_wrap",
            str(self.ring_size),
            "-reset_timestamps",
            "1",
            "-segment_format",
            "mpegts",
            str(self.ring_dir / "seg%03d.ts"),
        ]

    def _prepare_ring(self) -> None:
        self.ring_dir.mkdir(parents=True, exist_ok=True)
        for old in self.ring_dir.glob(self.SEGMENT_GLOB):
            old.unlink(missing_ok=True)
        free = shutil.disk_usage(self.ring_dir).free
        if free < self.MIN_FREE_BYTES:
            logger.warning(
                f"Only {free // 2**20} MB free in the segment ring directory (need >= {self.MIN_FREE_BYTES // 2**20} MB). "
                "The ring may fill up and archiving fail; in Docker raise --shm-size."
            )

    async def _supervise_ffmpeg(self) -> None:
        """Keeps the ffmpeg archiver running, restarting it with a back-off when it exits."""
        delay = self.RESTART_DELAY_MIN
        while True:
            started = time.monotonic()
            try:
                self._ffmpeg = await asyncio.create_subprocess_exec(
                    *self.ffmpeg_command(),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                logger.info("Segment archiver (ffmpeg -c copy) started.")
                code = await self._ffmpeg.wait()
                logger.warning(f"Segment archiver exited with code {code}.")
                self._drop_truncated_segment()
            except OSError:
                logger.exception("Could not start the segment archiver (is ffmpeg installed?).")
            if time.monotonic() - started >= self.STABLE_RUN_SECONDS:
                delay = self.RESTART_DELAY_MIN
            logger.info(f"Restarting segment archiver in {delay:.0f}s.")
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.RESTART_DELAY_MAX)

    def _drop_truncated_segment(self) -> None:
        """Deletes the newest segment after an ffmpeg exit: it was being written and may be cut off mid-frame."""
        newest: tuple[float, Path] | None = None
        for path in self.ring_dir.glob(self.SEGMENT_GLOB):
            try:
                candidate = (path.stat().st_mtime, path)
            except OSError:
                continue
            if newest is None or candidate > newest:
                newest = candidate
        if newest is not None:
            newest[1].unlink(missing_ok=True)

    async def _stop_ffmpeg(self) -> None:
        if self._supervisor and not self._supervisor.done():
            self._supervisor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._supervisor
        proc = self._ffmpeg
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                proc.kill()
                await proc.wait()

    def _complete_segments(self) -> list[tuple[float, Path]]:
        """Returns (mtime, path) of finished segments, oldest first; the newest (still being written) is left out."""
        found = []
        for path in self.ring_dir.glob(self.SEGMENT_GLOB):
            try:
                found.append((path.stat().st_mtime, path))
            except OSError:
                continue  # overwritten by the ring between glob and stat
        found.sort()
        return found[:-1]

    def _select_since(self, after_mtime: float | None) -> list[tuple[float, Path]]:
        """Selects complete segments newer than `after_mtime` (or, without it, within the pre-fire window)."""
        segments = self._complete_segments()
        if after_mtime is not None:
            return [seg for seg in segments if seg[0] > after_mtime]
        cutoff = time.time() - self.buffer_seconds - self.segment_seconds
        return [seg for seg in segments if seg[0] >= cutoff]

    @staticmethod
    def _concat_blocking(paths: list[Path], out_path: str) -> bool:
        """Joins segments into one mp4 with ``-c copy``; returns True on success."""
        listing = Path(out_path).with_suffix(".txt")
        listing.write_text("".join(f"file '{p}'\n" for p in paths))
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
                + ["-c", "copy", out_path],
                capture_output=True,
                check=False,
            )
        finally:
            listing.unlink(missing_ok=True)
        if result.returncode != 0:
            logger.error(f"Segment concat failed: {result.stderr.decode(errors='ignore')[-300:]}")
            return False
        return True

    @staticmethod
    def _copy_blocking(paths: list[Path], dest_dir: Path) -> list[Path]:
        """Copies segments out of the ring (which keeps overwriting) into `dest_dir`."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for src in paths:
            try:
                dest = dest_dir / f"{int(src.stat().st_mtime * 1000)}_{src.name}"
                shutil.copy2(src, dest)
                copied.append(dest)
            except OSError:
                logger.warning("A segment was overwritten before it could be copied.")
        return copied

    async def _write_clip(self, parts: list[Path], event_dir: str, label: str) -> bool:
        """Joins `parts` into one clip. Returns True when the clip was written (or there was nothing to write)."""
        if not parts:
            return True
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = os.path.join(event_dir, f"{self.context.FILENAME_PREFIX}{label}_{timestamp}{self.context.FILENAME_SUFFIX}")
        loop = asyncio.get_running_loop()
        if await loop.run_in_executor(None, self._concat_blocking, parts, out):
            self.context.current_video_path = out
            self.context._publish_chunk_path(out)
            logger.info(f"Saved {label} clip {os.path.basename(out)} ({len(parts)} segments).")
            return True
        return False

    async def handle_fire_event(self, event_dir: str) -> None:
        """Writes the pre-fire clip and then the post-fire chunks, entirely from ring segments.

        Args:
            event_dir: The destination directory for the archived videos.
        """
        loop = asyncio.get_running_loop()
        staging = Path(event_dir) / ".parts"
        try:
            selected = await loop.run_in_executor(None, self._select_since, None)
            last_mtime = selected[-1][0] if selected else time.time()
            # Copy first (the ring keeps overwriting the oldest slots), then join from the stable copies.
            parts = await loop.run_in_executor(None, self._copy_blocking, [p for _, p in selected], staging)
            await self._write_clip(parts, event_dir, "pre-fire-buffer")
            for part in parts:
                part.unlink(missing_ok=True)
            logger.info("Pre-fire clip written. Collecting post-fire segments.")

            chunk_seconds = self.context.chunk_length_seconds
            tail = self.context.chunks_to_keep_after_fire * chunk_seconds
            pending: list[Path] = []
            pending_since = time.monotonic()
            tail_deadline: float | None = None if settings.video.record_during_fire else time.monotonic() + tail
            while True:
                await asyncio.sleep(self.COLLECT_INTERVAL_SECONDS)
                fresh = await loop.run_in_executor(None, self._select_since, last_mtime)
                if fresh:
                    last_mtime = fresh[-1][0]
                    pending += await loop.run_in_executor(None, self._copy_blocking, [p for _, p in fresh], staging)
                if tail_deadline is None and self.context.signal_handler.is_fire_extinguished():
                    logger.info("Fire extinguished signal received.")
                    tail_deadline = time.monotonic() + tail
                done = tail_deadline is not None and time.monotonic() >= tail_deadline
                if pending and (done or time.monotonic() - pending_since >= chunk_seconds):
                    if await self._write_clip(pending, event_dir, "post-fire"):
                        for part in pending:
                            part.unlink(missing_ok=True)
                        pending = []
                    elif done:
                        logger.error("Could not write the last post-fire clip; %d segments were discarded.", len(pending))
                    else:
                        logger.warning("Writing a post-fire clip failed; will retry with the next segments.")
                    pending_since = time.monotonic()
                if done:
                    break
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        logger.info("Post-fire recording from segments is complete.")

    async def _run_main_loop(self) -> None:
        """Runs the archiver and waits for fire events, like the memory strategy."""
        if not self.context.source_url:
            logger.error("Segments mode needs the stream URL; archiving is inactive.")
        elif shutil.which("ffmpeg") is None:
            logger.error("ffmpeg not found; segments archiving is inactive.")
        else:
            await asyncio.get_running_loop().run_in_executor(None, self._prepare_ring)
            self._supervisor = asyncio.create_task(self._supervise_ffmpeg())
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
                    except (OSError, shutil.Error, subprocess.SubprocessError):  # fmt: skip
                        logger.exception("Unexpected but recoverable error during segments fire event handling.")
                    finally:
                        self.context.reset_after_fire()
        except asyncio.CancelledError:
            logger.info("Segments strategy main loop was cancelled.")
            raise
        finally:
            await asyncio.shield(self._stop_ffmpeg())
