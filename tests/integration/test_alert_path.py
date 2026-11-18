"""End-to-end test of the alert path with real wiring (#188).

CV-positive frames -> ``StreamFireDetector`` debounce -> ``_signal_fire`` -> ``Application._queue_fire_event`` ->
action queue -> ``Application._action_manager`` -> ``OnceAction`` -> ``SendToDiscord``. Only the stream, the frame
analysis and the HTTP request are faked; every callback between them is the production one.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from pytest import MonkeyPatch

from is_goat_burning.app import Application
from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings
from is_goat_burning.fire_detection.detector_core import StreamFireDetector
from is_goat_burning.on_fire_actions import OnceAction
from is_goat_burning.on_fire_actions import SendToDiscord
from is_goat_burning.on_fire_actions.send_to_discord import UNVERIFIED_NOTE

FIRE_SEQUENCE = [True, True, True, False, False, True]


class _FakeStream:
    """A stream yielding fixed numpy frames, then ending."""

    framerate = 1.0
    skips_decode = False

    def __init__(self, count: int) -> None:
        self.count = count
        self.stop = AsyncMock()

    async def frames(self) -> AsyncGenerator[np.ndarray]:
        for i in range(self.count):
            frame = np.full((48, 64, 3), 10 + i, dtype=np.uint8)
            frame[:, :, 2] = 200  # reddish, so the JPEG is clearly non-empty
            yield frame
            await asyncio.sleep(0)


class _FakeDetector:
    """A detector replaying a fire/no-fire sequence; ``last_result_unverified`` is a real bool."""

    def __init__(self, sequence: list[bool], unverified: bool) -> None:
        self.sequence = list(sequence)
        self.last_result_unverified = unverified
        self.last_result_inconclusive = False

    async def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        return self.sequence.pop(0), frame


@pytest.fixture
def app_settings(monkeypatch: MonkeyPatch) -> Settings:
    """Application settings with Discord on, no cooldown and no debounce."""
    settings = Settings(
        source="mock://source",
        log_level="CRITICAL",
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=True, hooks=["https://discord.invalid/api/webhooks/1/token"], attach_frame=True),
        video=VideoSettings(save_video_chunks=False),
        alert_cooldown_seconds=0.0,
        fire_detected_debounce_seconds=0.0,
        fire_extinguished_debounce_seconds=0.0,
    )
    monkeypatch.setattr("is_goat_burning.app.settings", settings)
    monkeypatch.setattr("is_goat_burning.fire_detection.detector_core.settings", settings)
    return settings


@pytest.fixture
def webhook() -> Generator[AsyncMock]:
    """Replaces the HTTP request of ``SendToDiscord`` (the payload and image are captured)."""
    with patch.object(SendToDiscord, "send_to_webhook", new=AsyncMock(return_value=True)) as mock:
        yield mock


async def _run_alert_path(app: Application, unverified: bool) -> StreamFireDetector:
    """Drives frames through a real detector wired to the application's real callbacks."""
    detector = StreamFireDetector(
        on_fire_action=app._queue_fire_event,
        video_output=False,
        checks_per_second=None,
        on_extinguished_action=app._queue_extinguished_event,
    )
    detector.signal_handler = MagicMock()  # the singleton's events coordinate the video saver, not used here
    detector.stream = _FakeStream(len(FIRE_SEQUENCE))  # type: ignore[assignment]
    detector.fire_detector = _FakeDetector(FIRE_SEQUENCE, unverified)  # type: ignore[assignment]
    manager = asyncio.create_task(app._action_manager())
    try:
        await detector()
        await app.action_queue.join()
    finally:
        manager.cancel()
        await asyncio.gather(manager, return_exceptions=True)
    return detector


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("unverified", [False, True])
async def test_fire_frames_reach_discord_with_frame_and_unverified_flag(
    app_settings: Settings,  # noqa: ARG001
    webhook: AsyncMock,
    unverified: bool,
) -> None:
    """Fire, extinguish, fire: two Discord alerts, each with the JPEG of the triggering frame and the right text."""
    app = Application()
    assert isinstance(app.on_fire_action_handler, OnceAction)

    await _run_alert_path(app, unverified)

    assert webhook.await_count == 2  # fire -> extinguished (re-armed) -> fire
    for call in webhook.await_args_list:
        _session, _url, payload, image = call.args
        assert isinstance(image, bytes)
        assert image[:2] == b"\xff\xd8"  # JPEG
        assert (UNVERIFIED_NOTE in payload["content"]) is unverified


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("unverified", [False, True])
async def test_alert_handler_receives_ndarray_frame_and_unverified(
    app_settings: Settings,  # noqa: ARG001
    unverified: bool,
) -> None:
    """The handler built by the application gets ``frame=<ndarray>`` and ``unverified=`` from the detector."""
    app = Application()
    received: list[dict[str, Any]] = []

    async def action(**kwargs: Any) -> None:
        received.append(kwargs)

    app.on_fire_action_handler = OnceAction([(action, {})], cooldown_seconds=0.0)

    await _run_alert_path(app, unverified)

    assert len(received) == 2
    for kwargs in received:
        assert isinstance(kwargs["frame"], np.ndarray)
        assert kwargs.get("unverified", False) is unverified
