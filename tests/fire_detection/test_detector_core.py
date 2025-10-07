"""Unit tests for the StreamFireDetector class.

These tests focus on the debouncing and state transition logic within the
detector's core, ensuring that signals are emitted correctly based on the
configured time delays.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from is_goat_burning.fire_detection.detector_core import StreamFireDetector


@pytest.fixture
def mock_settings() -> Generator[MagicMock, None, None]:
    """Fixture to mock the global settings object for debouncing."""
    with patch("is_goat_burning.fire_detection.detector_core.settings") as mock:
        mock.fire_detected_debounce_seconds = 1.0
        mock.fire_extinguished_debounce_seconds = 2.0
        yield mock


@pytest.fixture
def detector(mock_settings: MagicMock) -> StreamFireDetector:  # noqa: ARG001
    """Provides a StreamFireDetector instance with mocked dependencies."""
    mock_on_fire_action = AsyncMock()
    detector = StreamFireDetector(
        on_fire_action=mock_on_fire_action,
        video_output=False,
        checks_per_second=1.0,
    )
    detector.signal_handler = MagicMock()
    return detector


@pytest.mark.asyncio
async def test_fire_detected_debounce_prevents_signal_on_brief_detection(detector: StreamFireDetector) -> None:
    """Verify that a brief fire detection does not trigger a signal if it's shorter than the debounce time."""
    with patch("time.monotonic", side_effect=[100, 100.5]):  # 0.5s elapsed
        await detector._handle_fire_detection(fire_in_frame=True)  # Potential start
        await detector._handle_fire_detection(fire_in_frame=False)  # Fire gone before debounce ends

    detector.signal_handler.fire_detected.assert_not_called()
    assert detector.fire_is_currently_detected is False


@pytest.mark.asyncio
async def test_fire_detected_debounce_triggers_signal_after_delay(detector: StreamFireDetector) -> None:
    """Verify that a sustained fire detection triggers a signal after the debounce delay."""
    with patch("time.monotonic", side_effect=[100, 101.1]):  # 1.1s elapsed
        await detector._handle_fire_detection(fire_in_frame=True)  # Potential start
        await detector._handle_fire_detection(fire_in_frame=True)  # Debounce time exceeded

    detector.signal_handler.fire_detected.assert_called_once()
    assert detector.fire_is_currently_detected is True
    assert detector.on_fire_action.called is True


@pytest.mark.asyncio
async def test_fire_extinguished_debounce_prevents_signal_on_brief_absence(detector: StreamFireDetector) -> None:
    """Verify that a brief absence of fire does not trigger the extinguished signal."""
    detector.fire_is_currently_detected = True  # Assume fire is ongoing

    with patch("time.monotonic", side_effect=[200, 201.0]):  # 1.0s elapsed
        await detector._handle_fire_detection(fire_in_frame=False)  # Potential end
        await detector._handle_fire_detection(fire_in_frame=True)  # Fire returns before debounce

    detector.signal_handler.fire_extinguished.assert_not_called()
    assert detector.fire_is_currently_detected is True


@pytest.mark.asyncio
async def test_fire_extinguished_debounce_triggers_signal_after_delay(detector: StreamFireDetector) -> None:
    """Verify that a sustained absence of fire triggers the extinguished signal after the debounce delay."""
    detector.fire_is_currently_detected = True  # Assume fire is ongoing

    with patch("time.monotonic", side_effect=[200, 202.1]):  # 2.1s elapsed
        await detector._handle_fire_detection(fire_in_frame=False)  # Potential end
        await detector._handle_fire_detection(fire_in_frame=False)  # Debounce time exceeded

    detector.signal_handler.fire_extinguished.assert_called_once()
    assert detector.fire_is_currently_detected is False
