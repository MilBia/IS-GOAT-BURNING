"""Unit tests for the SendVideoToDiscord action class."""

import logging
from unittest.mock import MagicMock
from unittest.mock import patch

from aiohttp import FormData
import pytest

from is_goat_burning.logger import get_logger
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
        mock_post.return_value = _ok_response()
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
        mock_post.return_value = _ok_response()
        sender = SendVideoToDiscord(webhooks=[WEBHOOK, second])

        await sender("/videos/event/goat-cam_chunk.mp4")

    assert mock_post.call_count == 2


MIB = 1024 * 1024


def _ok_response(status: int = 200) -> MagicMock:
    """Builds a mocked `session.post` context manager returning the given status."""
    ctx = MagicMock()
    ctx.__aenter__.return_value.status = status
    ctx.__aenter__.return_value.raise_for_status = MagicMock()
    return ctx


@pytest.mark.asyncio
async def test_oversized_file_sends_substitute_without_upload() -> None:
    """Verifies a file above the configured limit is not read/uploaded; a text substitute with the file name is sent."""
    file_path = "/videos/event/goat-cam_big.mp4"
    with (
        patch("os.path.getsize", return_value=19 * MIB),
        patch.object(SendVideoToDiscord, "_read_file") as mock_read,
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
    ):
        mock_post.return_value = _ok_response()
        sender = SendVideoToDiscord(webhooks=[WEBHOOK], file_limit_mb=18)

        await sender(file_path)

    mock_read.assert_not_called()
    mock_post.assert_called_once()
    kwargs = mock_post.call_args.kwargs
    assert "data" not in kwargs
    content = kwargs["json"]["content"]
    assert "goat-cam_big.mp4" in content
    assert "19.0 MiB" in content
    assert "saved on disk" in content
    assert "/videos" not in content


@pytest.mark.asyncio
async def test_file_at_limit_is_uploaded() -> None:
    """Verifies a file exactly at the limit is still uploaded."""
    with (
        patch("os.path.getsize", return_value=18 * MIB),
        patch.object(SendVideoToDiscord, "_read_file", return_value=b"v"),
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
    ):
        mock_post.return_value = _ok_response()
        await SendVideoToDiscord(webhooks=[WEBHOOK], file_limit_mb=18)("/videos/e/a.mp4")

    assert isinstance(mock_post.call_args.kwargs["data"], FormData)


@pytest.mark.asyncio
async def test_http_413_sends_substitute_message(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies a 413 from Discord triggers the substitute message and one WARNING without the URL."""
    logger = get_logger("DiscordVideoSender")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger=logger.name)
    try:
        with (
            patch("os.path.getsize", return_value=10 * MIB),
            patch.object(SendVideoToDiscord, "_read_file", return_value=b"v"),
            patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
        ):
            mock_post.side_effect = [_ok_response(413), _ok_response(200)]
            await SendVideoToDiscord(webhooks=[WEBHOOK], file_limit_mb=18)("/videos/e/a.mp4")
    finally:
        logger.removeHandler(caplog.handler)

    assert mock_post.call_count == 2
    assert isinstance(mock_post.call_args_list[0].kwargs["data"], FormData)
    content = mock_post.call_args_list[1].kwargs["json"]["content"]
    assert "a.mp4" in content
    assert "10.0 MiB" in content
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "413" in warnings[0].getMessage()
    assert "abc" not in caplog.text
    assert "/webhooks/" not in caplog.text


@pytest.mark.asyncio
async def test_failure_logs_do_not_contain_webhook_url(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies logged upload errors never include the webhook URL (aiohttp errors embed it)."""
    from aiohttp import ClientResponseError  # noqa: PLC0415
    from aiohttp.client_reqrep import RequestInfo  # noqa: PLC0415
    from multidict import CIMultiDict  # noqa: PLC0415
    from yarl import URL  # noqa: PLC0415

    info = RequestInfo(URL(WEBHOOK), "POST", CIMultiDict(), URL(WEBHOOK))
    ctx = _ok_response()
    ctx.__aenter__.return_value.raise_for_status.side_effect = ClientResponseError(info, (), status=500, message="boom")
    logger = get_logger("DiscordVideoSender")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger=logger.name)
    try:
        with (
            patch("os.path.getsize", return_value=MIB),
            patch.object(SendVideoToDiscord, "_read_file", return_value=b"v"),
            patch("aiohttp.ClientSession.post", new_callable=MagicMock, return_value=ctx),
        ):
            await SendVideoToDiscord(webhooks=[WEBHOOK])("/videos/e/a.mp4")
    finally:
        logger.removeHandler(caplog.handler)

    assert "Failed to upload" in caplog.text
    assert "abc" not in caplog.text
    assert "/webhooks/" not in caplog.text


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
async def test_read_failure_after_stat_sends_nothing() -> None:
    """Verifies a file vanishing between the stat and the read sends no request."""
    with (
        patch("os.path.getsize", return_value=1024),
        patch.object(SendVideoToDiscord, "_read_file", side_effect=OSError("gone")),
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
    ):
        sender = SendVideoToDiscord(webhooks=[WEBHOOK])

        await sender("/videos/event/goat-cam_chunk.mp4")

    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_upload_continues_when_one_webhook_fails() -> None:
    """Verifies one failing webhook does not prevent uploads to the others."""
    second = "https://discord.com/api/webhooks/456/def"
    successful_post = _ok_response()

    with (
        patch("os.path.getsize", return_value=1024),
        patch.object(SendVideoToDiscord, "_read_file", return_value=b"video-bytes"),
        patch("aiohttp.ClientSession.post", new_callable=MagicMock) as mock_post,
    ):
        mock_post.side_effect = [Exception("Test Exception"), successful_post]
        sender = SendVideoToDiscord(webhooks=[WEBHOOK, second])

        await sender("/videos/event/goat-cam_chunk.mp4")

    assert mock_post.call_count == 2
