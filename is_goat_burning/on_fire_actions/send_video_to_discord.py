"""Provides a class to upload saved video chunks to Discord webhooks."""

import asyncio
from dataclasses import dataclass
from dataclasses import field
import os
from urllib.parse import urlparse

from aiohttp import ClientSession
from aiohttp import ClientTimeout
from aiohttp import FormData

from is_goat_burning.logger import get_logger

logger = get_logger("DiscordVideoSender")

# Discord webhooks reject uploads above ~25 MB. A slightly lower ceiling is used
# to leave headroom for the multipart envelope and avoid borderline rejections.
DISCORD_FILE_LIMIT = 24 * 1024 * 1024
# Discord truncates message content beyond this many characters.
DISCORD_CHARACTER_LIMIT = 2000
# Timeout for the small fallback JSON message sent when a file is oversized.
FALLBACK_TIMEOUT_SECONDS = 3.0


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class SendVideoToDiscord:
    """An awaitable class that uploads a video file to one or more Discord webhooks.

    This action is designed to be driven by a `FireEventAction` container, whose
    background task calls it once per archived video chunk with the chunk's file
    path. If the file is within Discord's size limit it is uploaded as a
    multipart attachment; otherwise a fallback text message pointing to the local
    file is sent instead.

    Attributes:
        webhooks (list[str]): A list of Discord webhook URLs to upload to.
        upload_timeout_seconds (float): The total timeout for a single upload
            request, in seconds.
        oversized_message_template (str): The message sent when a file exceeds
            the Discord upload limit. It must contain a ``{file_path}``
            placeholder.
    """

    webhooks: list[str]
    upload_timeout_seconds: float = 60.0
    oversized_message_template: str = field(
        default=("Goat on fire! A video chunk was saved but is too large to upload to Discord. You can find it at: {file_path}")
    )

    @staticmethod
    def _read_file(file_path: str) -> bytes:
        """Reads a file's contents in binary mode.

        Args:
            file_path: The absolute path of the file to read.

        Returns:
            The raw bytes of the file.
        """
        with open(file_path, "rb") as file:
            return file.read()

    async def _post_file(self, session: ClientSession, url: str, data: bytes, filename: str) -> None:
        """Uploads the video file to a single Discord webhook URL.

        Args:
            session: The `aiohttp.ClientSession` to use for the request.
            url: The specific webhook URL to upload to.
            data: The raw bytes of the video file.
            filename: The filename to present to Discord.

        Raises:
            aiohttp.ClientError: If a non-2xx status code is received.
        """
        form = FormData()
        form.add_field("file", data, filename=filename, content_type="video/mp4")
        async with session.post(url=url, data=form, timeout=ClientTimeout(total=self.upload_timeout_seconds)) as response:
            response.raise_for_status()

    async def _post_json(self, session: ClientSession, url: str, content: str) -> None:
        """Sends a fallback JSON message to a single Discord webhook URL.

        Args:
            session: The `aiohttp.ClientSession` to use for the request.
            url: The specific webhook URL to send the message to.
            content: The message content.

        Raises:
            aiohttp.ClientError: If a non-2xx status code is received.
        """
        async with session.post(
            url=url,
            json={"content": content[:DISCORD_CHARACTER_LIMIT]},
            headers={"User-Agent": "Python/3", "Content-Type": "application/json"},
            timeout=ClientTimeout(total=FALLBACK_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()

    def _log_results(self, results: list[object], verb: str) -> None:
        """Logs the outcome of the per-webhook requests.

        Args:
            results: The list of results (or exceptions) from `asyncio.gather`.
            verb: A short description of the operation for the log message.
        """
        success_count = 0
        for url, result in zip(self.webhooks, results, strict=True):
            if isinstance(result, Exception):
                sanitized_host = urlparse(url).hostname or "invalid-host"
                logger.error(f"Failed to {verb} to webhook {sanitized_host}: [{result.__class__.__name__}] {result}")
            else:
                success_count += 1
        logger.info(f"Video chunk {verb} complete. Successful: {success_count}/{len(self.webhooks)}.")

    async def _upload_file(self, session: ClientSession, file_path: str) -> None:
        """Reads and uploads the file concurrently to all configured webhooks."""
        try:
            data = await asyncio.to_thread(self._read_file, file_path)
        except OSError as e:
            logger.error(f"Could not read video chunk {file_path}: [{e.__class__.__name__}] {e}")
            return
        filename = os.path.basename(file_path)
        logger.info(f"Uploading video chunk {filename} to {len(self.webhooks)} webhook(s)...")
        tasks = [self._post_file(session, url, data, filename) for url in self.webhooks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self._log_results(results, "upload")

    async def _send_oversized_fallback(self, session: ClientSession, file_path: str) -> None:
        """Sends the oversized-file fallback message to all configured webhooks."""
        content = self.oversized_message_template.format(file_path=file_path)
        logger.info("Video chunk exceeds the Discord upload limit; sending fallback message.")
        tasks = [self._post_json(session, url, content) for url in self.webhooks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self._log_results(results, "fallback")

    async def __call__(self, file_path: str) -> None:
        """Uploads the given video chunk to Discord, or sends a fallback message.

        Args:
            file_path: The absolute path of the saved video chunk to send.
        """
        try:
            file_size = await asyncio.to_thread(os.path.getsize, file_path)
        except OSError as e:
            logger.error(f"Could not stat video chunk {file_path}: [{e.__class__.__name__}] {e}")
            return

        async with ClientSession() as session:
            if file_size < DISCORD_FILE_LIMIT:
                await self._upload_file(session, file_path)
            else:
                await self._send_oversized_fallback(session, file_path)
