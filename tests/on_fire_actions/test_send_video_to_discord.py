"""Unit tests for the SendVideoToDiscord action class."""

from unittest.mock import MagicMock
from unittest.mock import patch

from aiohttp import FormData
import pytest

from is_goat_burning.on_fire_actions.send_video_to_discord import DISCORD_FILE_LIMIT
from is_goat_burning.on_fire_actions.send_video_to_discord import SendVideoToDiscord

WEBHOOK = "https://discord.com/api/webhooks/123/abc"


@pytest.mark.asyncio
async def test_uploads_small_file_as_multipart() -> None:
    """Verifies a file under the limit is uploaded via multipart form data."""
    with (
        patch("os.path.getsize", return_value=1024),
        patch.object(SendVideoToDiscord, "_read_file", return_value=b"video-bytes"),
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
    ):
        mock_post.return_value.__aenter__.return_value.raise_for_status = MagicMock()
        sender = SendVideoToDiscord(webhooks=[WEBHOOK])

        await sender("/videos/event/goat-cam_chunk.mp4")

    mock_post.assert_called_once()
    kwargs = mock_post.call_args.kwargs
    assert kwargs["url"] == WEBHOOK
    assert isinstance(kwargs["data"], FormData)
    assert "json" not in kwargs


@pytest.mark.asyncio
async def test_uploads_to_all_webhooks() -> None:
    """Verifies the file is uploaded to every configured webhook."""
    second = "https://discord.com/api/webhooks/456/def"
    with (
        patch("os.path.getsize", return_value=1024),
        patch.object(SendVideoToDiscord, "_read_file", return_value=b"video-bytes"),
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
    ):
        mock_post.return_value.__aenter__.return_value.raise_for_status = MagicMock()
        sender = SendVideoToDiscord(webhooks=[WEBHOOK, second])

        await sender("/videos/event/goat-cam_chunk.mp4")

    assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_oversized_file_sends_fallback_message() -> None:
    """Verifies an oversized file triggers the JSON fallback message."""
    file_path = "/videos/event/goat-cam_big.mp4"
    with (
        patch("os.path.getsize", return_value=DISCORD_FILE_LIMIT + 1),
        patch.object(SendVideoToDiscord, "_read_file") as mock_read,
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
    ):
        mock_post.return_value.__aenter__.return_value.raise_for_status = MagicMock()
        sender = SendVideoToDiscord(webhooks=[WEBHOOK])

        await sender(file_path)

    mock_read.assert_not_called()
    mock_post.assert_called_once()
    kwargs = mock_post.call_args.kwargs
    assert "data" not in kwargs
    content = kwargs["json"]["content"]
    assert content == (
        f"Goat on fire! A video chunk was saved but is too large to upload to Discord. You can find it at: {file_path}"
    )


@pytest.mark.asyncio
async def test_missing_file_sends_nothing() -> None:
    """Verifies that a stat failure results in no request being sent."""
    with (
        patch("os.path.getsize", side_effect=OSError("missing")),
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
    ):
        sender = SendVideoToDiscord(webhooks=[WEBHOOK])

        await sender("/videos/event/gone.mp4")

    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_upload_continues_when_one_webhook_fails() -> None:
    """Verifies one failing webhook does not prevent uploads to the others."""
    second = "https://discord.com/api/webhooks/456/def"
    successful_post = MagicMock()
    successful_post.__aenter__.return_value.raise_for_status = MagicMock()

    with (
        patch("os.path.getsize", return_value=1024),
        patch.object(SendVideoToDiscord, "_read_file", return_value=b"video-bytes"),
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
    ):
        mock_post.side_effect = [Exception("Test Exception"), successful_post]
        sender = SendVideoToDiscord(webhooks=[WEBHOOK, second])

        await sender("/videos/event/goat-cam_chunk.mp4")

    assert mock_post.call_count == 2
