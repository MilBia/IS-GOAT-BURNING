"""Unit tests for the SendToDiscord action class."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from is_goat_burning.on_fire_actions.send_to_discord import SendToDiscord


@pytest.mark.asyncio
async def test_send_to_discord_success() -> None:
    """Verifies a successful message send to a single webhook."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        # Mock the async context manager and the response method
        mock_post.return_value.__aenter__.return_value.raise_for_status = MagicMock()
        discord_sender = SendToDiscord(
            webhooks=["https://discord.com/api/webhooks/123/abc"],
            message="Test Message",
        )
        await discord_sender()

        # Assert that the post method was called with the correct data
        mock_post.assert_called_once_with(
            url="https://discord.com/api/webhooks/123/abc",
            json={"content": "Test Message"},
            headers={"User-Agent": "Python/3", "Content-Type": "application/json"},
            timeout=3,
        )


@pytest.mark.asyncio
async def test_send_to_discord_multiple_webhooks() -> None:
    """Verifies that multiple webhooks are all called."""
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.raise_for_status = MagicMock()
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
    successful_post_mock.__aenter__.return_value.raise_for_status = MagicMock()

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
