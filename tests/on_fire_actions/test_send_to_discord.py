from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from is_goat_burning.on_fire_actions.send_to_discord import SendToDiscord


@pytest.mark.asyncio
async def test_send_to_discord_success():
    """
    Tests that the SendToDiscord class can successfully send a message to a
    single webhook.
    """
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.raise_for_status = MagicMock()
        discord_sender = SendToDiscord(
            webhooks=["https://discord.com/api/webhooks/123/abc"],
            message="Test Message",
        )
        await discord_sender()

        mock_post.assert_called_once_with(
            url="https://discord.com/api/webhooks/123/abc",
            json={"content": "Test Message"},
            headers={"User-Agent": "Python/3", "Content-Type": "application/json"},
            timeout=3,
        )


@pytest.mark.asyncio
async def test_send_to_discord_multiple_webhooks():
    """
    Tests that the SendToDiscord class can successfully send a message to
    multiple webhooks.
    """
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.raise_for_status = MagicMock()
        discord_sender = SendToDiscord(
            webhooks=["https://discord.com/api/webhooks/123/abc", "https://discord.com/api/webhooks/456/def"],
            message="Test Message",
        )
        await discord_sender()

        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_send_to_discord_one_webhook_fails():
    """
    Tests that the SendToDiscord class can handle a single webhook failing.
    """
    successful_post_mock = MagicMock()
    successful_post_mock.__aenter__.return_value.raise_for_status = MagicMock()
    with patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post:
        mock_post.side_effect = [Exception("Test Exception"), successful_post_mock]
        discord_sender = SendToDiscord(
            webhooks=["https://discord.com/api/webhooks/123/abc", "https://discord.com/api/webhooks/456/def"],
            message="Test Message",
        )
        await discord_sender()
        assert mock_post.call_count == 2
