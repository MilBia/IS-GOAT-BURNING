"""Unit tests for the SendToDiscord action class."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from aiohttp import ClientTimeout
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
        await discord_sender()
        # Assert that the client still attempted to call both webhooks
        assert mock_post.call_count == 2


URL = "https://discord.com/api/webhooks/123/abc"


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
        assert await _sender(max_attempts=3)() is False
    assert mock_post.call_count == 3
    assert "abc" not in caplog.text
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
