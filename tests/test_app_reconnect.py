"""Tests for the reconnect loop of `Application`: exponential backoff, quiet logging, first-frame connect signal."""

import asyncio
import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pytest import LogCaptureFixture
from pytest import MonkeyPatch
from yt_dlp.utils import DownloadError

from is_goat_burning.app import Application
from is_goat_burning.config import DiscordSettings
from is_goat_burning.config import EmailSettings
from is_goat_burning.config import Settings
from is_goat_burning.config import VideoSettings
from is_goat_burning.service_notifier import ServiceNotifier

BASE_DELAY = 5
MAX_DELAY = 40
ATTEMPTS = 7
SERVICE_THRESHOLD = 600.0
TOKEN_URL = "https://manifest.example.com/video?token=SECRET123"


@pytest.fixture
def test_settings(monkeypatch: MonkeyPatch) -> Settings:
    """Settings with a small backoff range."""
    settings = Settings(
        source="mock://source",
        log_level="CRITICAL",
        reconnect_delay_seconds=BASE_DELAY,
        reconnect_max_delay_seconds=MAX_DELAY,
        email=EmailSettings(use_emails=False),
        discord=DiscordSettings(use_discord=False),
        video=VideoSettings(save_video_chunks=False),
    )
    monkeypatch.setattr("is_goat_burning.app.settings", settings)
    return settings


def _app(attempts: int) -> tuple[Application, list[float]]:
    """An application whose loop stops after `attempts` sleeps; returns it with the recorded delays."""
    app = Application()
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= attempts:
            app.signal_handler.is_running.return_value = False

    app._backoff_sleep = fake_sleep  # type: ignore[method-assign]
    return app, delays


@pytest.fixture
def app_log(caplog: LogCaptureFixture) -> LogCaptureFixture:
    """Captures the 'Application' logger (it does not propagate, so the handler is attached directly)."""
    logger = logging.getLogger("Application")
    logger.addHandler(caplog.handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    yield caplog
    logger.removeHandler(caplog.handler)
    logger.setLevel(old_level)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_unresolvable_url_backs_off_exponentially_up_to_the_cap(test_settings: Settings) -> None:  # noqa: ARG001
    """Consecutive failures wait 5, 10, 20, 40, 40, ... seconds."""
    app, delays = _app(ATTEMPTS)
    err = ValueError("Failed to resolve stream URL.")
    with patch("is_goat_burning.app.StreamFireDetector.create", new=AsyncMock(side_effect=err)):
        await app.run()
    await app.shutdown()
    assert delays == [5, 10, 20, 40, 40, 40, 40]


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_known_failures_log_one_warning_line_without_traceback_or_url(
    test_settings: Settings,  # noqa: ARG001
    app_log: LogCaptureFixture,
) -> None:
    """A resolution failure is one WARNING line (type, short message, next delay): no traceback, no tokenized URL."""
    app, _ = _app(2)
    cause = DownloadError(f"ERROR: [youtube] abc: This live event will begin in 2 days {TOKEN_URL}")
    err = ValueError("Failed to resolve stream URL.")
    err.__cause__ = cause
    with patch("is_goat_burning.app.StreamFireDetector.create", new=AsyncMock(side_effect=err)):
        await app.run()
    await app.shutdown()
    failures = [r for r in app_log.records if r.levelno >= logging.WARNING]
    assert len(failures) == 2
    for record in failures:
        assert record.levelno == logging.WARNING
        assert record.exc_info is None
        assert "SECRET123" not in record.getMessage()
        assert "ValueError" in record.getMessage()
        assert "live event will begin" in record.getMessage()
    assert "5s" in failures[0].getMessage()
    assert "10s" in failures[1].getMessage()
    assert "Traceback" not in app_log.text


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_unknown_failure_keeps_the_traceback_but_still_backs_off(
    test_settings: Settings,  # noqa: ARG001
    app_log: LogCaptureFixture,
) -> None:
    """An unexpected exception is logged at ERROR with its traceback and uses the same backoff."""
    app, delays = _app(2)
    with patch("is_goat_burning.app.StreamFireDetector.create", new=AsyncMock(side_effect=ZeroDivisionError("boom"))):
        await app.run()
    await app.shutdown()
    errors = [r for r in app_log.records if r.levelno == logging.ERROR]
    assert len(errors) == 2
    assert all(r.exc_info for r in errors)
    assert delays == [5, 10]


class _Clock:
    """A manually advanced monotonic clock for the session-duration check."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.mark.asyncio
@pytest.mark.timeout(5)
@pytest.mark.parametrize(("session_seconds", "expected"), [(60.0, [5, 10, 5, 10]), (59.0, [5, 10, 20, 40])])
async def test_delay_resets_only_after_a_long_enough_session(
    test_settings: Settings,  # noqa: ARG001
    monkeypatch: MonkeyPatch,
    session_seconds: float,
    expected: list[float],
) -> None:
    """Failures double the delay; a session with a first frame resets it only if it lasted at least 60 s (#188).

    A stream that yields one frame and dies at once must keep backing off.
    """
    clock = _Clock()
    monkeypatch.setattr("is_goat_burning.app.time", clock)
    app, delays = _app(4)
    app.service_notifier = MagicMock(stop=AsyncMock())
    calls = {"n": 0}

    async def create(**kwargs: object) -> AsyncMock:
        calls["n"] += 1
        if calls["n"] != 3:  # noqa: PLR2004
            raise ConnectionError("down")
        detector = AsyncMock()
        detector.fire_is_currently_detected = False

        async def run() -> None:
            kwargs["on_first_frame"]()  # type: ignore[operator]
            clock.now += session_seconds
            raise ConnectionError("died later")

        detector.side_effect = run
        return detector

    with patch("is_goat_burning.app.StreamFireDetector.create", new=create):
        await app.run()
    await app.shutdown()
    # fail, fail, (session with a frame) -> reset if long enough, fail again.
    assert delays == expected


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_stream_connected_is_signalled_only_after_the_first_frame(test_settings: Settings) -> None:  # noqa: ARG001
    """A stream that dies before any frame never calls ``stream_connected``; one with a frame does."""
    app, _ = _app(2)
    app.service_notifier = MagicMock(stop=AsyncMock())
    sessions = iter([False, True])

    async def create(**kwargs: object) -> AsyncMock:
        got_frame = next(sessions)
        detector = AsyncMock()
        detector.fire_is_currently_detected = False

        async def run() -> None:
            app.service_notifier.stream_connected.assert_not_called()
            if got_frame:
                kwargs["on_first_frame"]()  # type: ignore[operator]
                app.service_notifier.stream_connected.assert_called_once()
            raise ConnectionError("died")

        detector.side_effect = run
        return detector

    with patch("is_goat_burning.app.StreamFireDetector.create", new=create):
        await app.run()
    await app.shutdown()
    assert app.service_notifier.stream_connected.call_count == 1
    assert app.service_notifier.stream_failed.call_count == 2


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_flapping_stream_is_reported_as_an_outage(test_settings: Settings) -> None:  # noqa: ARG001
    """A stream that connects but never delivers a frame keeps the outage open until the warning threshold."""
    now = {"t": 0.0}
    send = AsyncMock(return_value=True)
    notifier = ServiceNotifier(
        ["https://hook.invalid/x"],
        strategy="classic",
        source="mock://source",
        stream_down_threshold=SERVICE_THRESHOLD,
        send=send,
        clock=lambda: now["t"],
    )
    app = Application()
    app.service_notifier = notifier
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True
    cycles = {"n": 0}

    async def fake_sleep(delay: float) -> None:
        now["t"] += delay + 2  # the stream lived ~2 s before it died
        cycles["n"] += 1
        if cycles["n"] >= 40:  # noqa: PLR2004
            app.signal_handler.is_running.return_value = False

    app._backoff_sleep = fake_sleep  # type: ignore[method-assign]

    async def create(**_kwargs: object) -> AsyncMock:
        detector = AsyncMock()
        detector.fire_is_currently_detected = False
        detector.side_effect = ConnectionError("died right after connect")
        return detector

    with patch("is_goat_burning.app.StreamFireDetector.create", new=create):
        await app.run()
    notifier.check_timers()
    await notifier.drain()
    await app.shutdown()
    texts = [c.args[0] for c in send.await_args_list]
    assert any("unavailable for" in t for t in texts)
    assert not any("recovered" in t or "connected after" in t for t in texts)


@pytest.mark.asyncio
async def test_backoff_sleep_returns_promptly_on_shutdown(test_settings: Settings) -> None:  # noqa: ARG001
    """A long backoff sleep ends within about a second once the signal handler stops running."""
    app = Application()
    app.signal_handler = MagicMock()
    app.signal_handler.is_running.return_value = True
    task = asyncio.create_task(app._backoff_sleep(300))
    await asyncio.sleep(0.05)
    app.signal_handler.is_running.return_value = False
    await asyncio.wait_for(task, timeout=2.5)
