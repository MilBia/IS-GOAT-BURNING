"""Tests for the small #133 optimizations: bounded frame queue and skipping the unused annotated frame."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from is_goat_burning.fire_detection.detectors import CPUFireDetector
from is_goat_burning.stream_recording.save_stream_to_file import AsyncVideoChunkSaver
from is_goat_burning.stream_recording.strategies import DiskBufferStrategy

LOWER = np.array([18, 50, 50], dtype="uint8")
UPPER = np.array([35, 255, 255], dtype="uint8")


@pytest.mark.asyncio
async def test_frame_queue_is_bounded_and_drops_newest_frames() -> None:
    saver = AsyncVideoChunkSaver(
        enabled=False, output_dir="x", chunk_length_seconds=1, max_chunks=1, chunks_to_keep_after_fire=1
    )
    assert saver.frame_queue.maxsize == AsyncVideoChunkSaver.FRAME_QUEUE_MAX_FRAMES
    strategy = DiskBufferStrategy(MagicMock(frame_queue=saver.frame_queue))
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    for _ in range(AsyncVideoChunkSaver.FRAME_QUEUE_MAX_FRAMES + 25):
        strategy.add_frame(frame)  # must not raise or grow past the bound

    assert saver.frame_queue.qsize() == AsyncVideoChunkSaver.FRAME_QUEUE_MAX_FRAMES
    assert isinstance(saver.frame_queue, asyncio.Queue)


def _moving_pair() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((40, 40, 3), dtype=np.uint8)
    b = a.copy()
    b[:, :, 2] = 255
    b[:, :, 1] = 190  # orange-yellow in BGR, appears between the two frames
    return a, b


def test_annotation_off_returns_input_frame_with_same_verdict() -> None:
    a, b = _moving_pair()
    on, off = CPUFireDetector(10, LOWER, UPPER), CPUFireDetector(10, LOWER, UPPER)
    off.annotate = False

    on._detect_logic(a)
    off._detect_logic(a)
    fire_on, annotated = on._detect_logic(b)
    fire_off, passthrough = off._detect_logic(b)

    assert fire_on is True and fire_off is True
    assert passthrough is b
    assert annotated is not b


@pytest.mark.asyncio
async def test_stream_detector_disables_annotation_without_video_output(mocker: MockerFixture) -> None:
    from is_goat_burning.fire_detection import detector_core

    stream = MagicMock(frame_shape=(10, 10))
    mocker.patch.object(detector_core.YouTubeStream, "resolve_url", mocker.AsyncMock(return_value="u"))
    mocker.patch.object(detector_core.VideoStreamer, "create", mocker.AsyncMock(return_value=stream))
    local = CPUFireDetector(1, LOWER, UPPER)
    mocker.patch.object(detector_core, "create_fire_detector", return_value=MagicMock(local_detector=local))

    await detector_core.StreamFireDetector.create(src="s", on_fire_action=mocker.AsyncMock(), video_output=False)
    assert local.annotate is False

    local.annotate = True
    await detector_core.StreamFireDetector.create(src="s", on_fire_action=mocker.AsyncMock(), video_output=True)
    assert local.annotate is True
