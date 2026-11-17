"""Unit tests for the SendToDiscord action class."""

from collections.abc import Iterator
import json
import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from aiohttp import ClientTimeout
from aiohttp import FormData
import cv2
import numpy as np
import pytest

from is_goat_burning.on_fire_actions.send_to_discord import SendToDiscord


@pytest.mark.asyncio
async def test_send_to_discord_success() -> None:
    """Verifies a successful message send to a single webhook."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        # Mock the async context manager and the response method
        mock_post.return_value.__aenter__.return_value.status = 204
        discord_sender = SendToDiscord(
            webhooks=["https://discord.com/api/webhooks/123/abc"],
            message="Test Message",
        )
        assert await discord_sender() is True

        # Assert that the post method was called with the correct data
        mock_post.assert_called_once_with(
            url="https://discord.com/api/webhooks/123/abc",
            json={"content": "Test Message"},
            headers={"User-Agent": "Python/3", "Content-Type": "application/json"},
            timeout=ClientTimeout(total=10.0),
        )


@pytest.mark.asyncio
async def test_send_to_discord_multiple_webhooks() -> None:
    """Verifies that multiple webhooks are all called."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 204
        discord_sender = SendToDiscord(
            webhooks=["https://discord.com/api/webhooks/123/abc", "https://discord.com/api/webhooks/456/def"],
            message="Test Message",
        )
        await discord_sender()

        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_send_to_discord_one_webhook_fails() -> None:
    """Verifies that one failed webhook does not prevent others from being called."""
    successful_post_mock = MagicMock()
    successful_post_mock.__aenter__.return_value.status = 204

    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        # Simulate one failure and one success
        mock_post.side_effect = [Exception("Test Exception"), successful_post_mock]
        discord_sender = SendToDiscord(
            webhooks=["https://discord.com/api/webhooks/123/abc", "https://discord.com/api/webhooks/456/def"],
            message="Test Message",
        )
        assert await discord_sender() is True
        # Assert that the client still attempted to call both webhooks
        assert mock_post.call_count == 2


URL = "https://discord.com/api/webhooks/123/abc"
SECRET_URL = "https://discord.com/api/webhooks/123/SECRETTOKEN"


@pytest.fixture(autouse=True)
def _capture_discord_logs(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Attaches caplog to the non-propagating DiscordSender logger."""
    discord_logger = logging.getLogger("DiscordSender")
    discord_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="DiscordSender")
    yield
    discord_logger.removeHandler(caplog.handler)


def _resp(status: int, headers: dict[str, str] | None = None, body: dict | None = None) -> MagicMock:
    """Builds a mock `session.post(...)` context manager yielding a response."""
    ctx = MagicMock()
    response = ctx.__aenter__.return_value
    response.status = status
    response.headers = headers or {}
    response.json = AsyncMock(return_value=body or {})
    return ctx


def _sender(**kwargs: object) -> SendToDiscord:
    return SendToDiscord(webhooks=[URL], message="m", **kwargs)


@pytest.mark.asyncio
async def test_retries_on_500_then_succeeds() -> None:
    """A 500 is retried with growing delays until the webhook accepts."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        mock_post.side_effect = [_resp(500), _resp(502), _resp(204)]
        assert await _sender(base_delay=1.0, max_delay=30.0)() is True
    assert mock_post.call_count == 3
    assert [c.args[0] for c in sleep.await_args_list] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_retries_on_429_honouring_retry_after_header() -> None:
    """A 429 waits for the Retry-After header value."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        mock_post.side_effect = [_resp(429, headers={"Retry-After": "7"}), _resp(204)]
        assert await _sender()() is True
    assert sleep.await_args_list[0].args[0] == 7.0


@pytest.mark.asyncio
async def test_429_uses_json_retry_after_and_caps_it() -> None:
    """Without the header the JSON `retry_after` is used, capped at max_delay."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        mock_post.side_effect = [_resp(429, body={"retry_after": 999.5}), _resp(204)]
        assert await _sender(max_delay=30.0)() is True
    assert sleep.await_args_list[0].args[0] == 30.0


@pytest.mark.asyncio
async def test_retries_on_timeout_then_succeeds() -> None:
    """A timeout is retried."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_post.side_effect = [TimeoutError(), _resp(204)]
        assert await _sender()() is True
    assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_does_not_retry_other_4xx() -> None:
    """A 404 is permanent: one attempt, action reports failure."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        mock_post.side_effect = [_resp(404)]
        assert await _sender()() is False
    assert mock_post.call_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_attempts_fail_returns_false_without_leaking_url(caplog: pytest.LogCaptureFixture) -> None:
    """Exhausted retries return False and never log the webhook URL."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_post.side_effect = lambda **_: _resp(500)
        assert await SendToDiscord(webhooks=[SECRET_URL], message="m", max_attempts=3)() is False
    assert mock_post.call_count == 3
    assert "discord.com" in caplog.text  # logs captured, host is shown
    assert "SECRETTOKEN" not in caplog.text
    assert "/api/webhooks" not in caplog.text


@pytest.mark.asyncio
async def test_partial_success_returns_true() -> None:
    """One webhook succeeding is enough for True even if the other fails permanently."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_post.side_effect = lambda **kw: _resp(204) if kw["url"].endswith("/ok") else _resp(404)
        sender = SendToDiscord(webhooks=["https://d.example/ok", "https://d.example/bad"], message="m")
        assert await sender() is True


@pytest.mark.asyncio
async def test_non_finite_retry_after_falls_back_to_backoff() -> None:
    """`Retry-After: nan` must not reach asyncio.sleep; backoff delay is used."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        mock_post.side_effect = [_resp(429, headers={"Retry-After": "nan"}), _resp(204)]
        assert await _sender(base_delay=2.0)() is True
    assert sleep.await_args_list[0].args[0] == 2.0


@pytest.mark.asyncio
async def test_exhausted_429_returns_false_without_final_sleep() -> None:
    """Persistent 429: sleeps between attempts only, then reports failure."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        mock_post.side_effect = lambda **_: _resp(429, headers={"Retry-After": "1"})
        assert await _sender(max_attempts=3)() is False
    assert mock_post.call_count == 3
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_unexpected_exception_log_omits_message(caplog: pytest.LogCaptureFixture) -> None:
    """An unexpected error whose text embeds the URL is logged by class name only."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.side_effect = RuntimeError(f"boom {SECRET_URL}")
        assert await SendToDiscord(webhooks=[SECRET_URL], message="m")() is False
    assert "RuntimeError" in caplog.text
    assert "SECRETTOKEN" not in caplog.text


# --- unverified alerts (#169) ---


@pytest.mark.asyncio
async def test_unverified_alert_carries_a_note() -> None:
    """An unverified alert appends a note so the recipient can tell it apart; the note survives truncation."""
    from is_goat_burning.on_fire_actions.send_to_discord import UNVERIFIED_NOTE

    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 204
        sender = SendToDiscord(webhooks=["https://discord.com/api/webhooks/1/a"], message="x" * 2500)
        assert await sender(unverified=True) is True
        content = mock_post.call_args.kwargs["json"]["content"]
        assert content.endswith(UNVERIFIED_NOTE)
        assert len(content) == 2000

        await sender()
        assert mock_post.call_args.kwargs["json"]["content"] == "x" * 2000


# --- frame attachment (#177) ---


def _frame(width: int = 64, height: int = 48) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)


def _form_fields(form: FormData) -> dict[str, tuple[dict[str, str], object]]:
    return {f[0]["name"]: (f[0], f[2]) for f in form._fields}


@pytest.mark.asyncio
async def test_frame_is_sent_as_multipart_jpeg() -> None:
    """With a frame the request is multipart: payload_json plus files[0] holding a JPEG."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 204
        assert await _sender()(frame=_frame()) is True
    kwargs = mock_post.call_args.kwargs
    assert "json" not in kwargs
    fields = _form_fields(kwargs["data"])
    assert json.loads(fields["payload_json"][1]) == {"content": "m"}
    meta, content = fields["files[0]"]
    assert meta["filename"] == "frame.jpg"
    assert bytes(content)[:2] == b"\xff\xd8"
    assert "application/json" not in kwargs["headers"].values()


@pytest.mark.asyncio
async def test_no_frame_sends_plain_json() -> None:
    """Without a frame the text alert is unchanged."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 204
        assert await _sender()() is True
    assert mock_post.call_args.kwargs["json"] == {"content": "m"}


@pytest.mark.asyncio
async def test_attach_frame_disabled_sends_text_only() -> None:
    """``attach_frame=False`` ignores the frame."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 204
        assert await _sender(attach_frame=False)(frame=_frame()) is True
    assert "json" in mock_post.call_args.kwargs


@pytest.mark.asyncio
async def test_encoding_failure_still_sends_the_text() -> None:
    """A frame that cannot be encoded never blocks the alert."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 204
        assert await _sender()(frame=np.zeros((0, 0, 3), dtype=np.uint8)) is True
    assert mock_post.call_args.kwargs["json"] == {"content": "m"}


@pytest.mark.asyncio
async def test_large_frame_is_downscaled() -> None:
    """Frames wider than 1280 px are downscaled before encoding."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 204
        await _sender()(frame=_frame(2560, 1440))
    content = _form_fields(mock_post.call_args.kwargs["data"])["files[0]"][1]
    decoded = cv2.imdecode(np.frombuffer(bytes(content), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 1280


@pytest.mark.asyncio
async def test_retry_rebuilds_the_form_with_the_image() -> None:
    """Every attempt gets a fresh FormData that still carries the image."""
    with (
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        patch("is_goat_burning.on_fire_actions.send_to_discord.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_post.side_effect = [_resp(500), _resp(204)]
        assert await _sender()(frame=_frame()) is True
    forms = [c.kwargs["data"] for c in mock_post.call_args_list]
    assert forms[0] is not forms[1]
    assert all("files[0]" in _form_fields(f) for f in forms)
