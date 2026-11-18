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

BYTES_PER_MIB = 1024 * 1024
HTTP_PAYLOAD_TOO_LARGE = 413
# Discord truncates message content beyond this many characters.
DISCORD_CHARACTER_LIMIT = 2000
# Timeout for the small substitute JSON message sent when a file cannot be uploaded.
FALLBACK_TIMEOUT_SECONDS = 3.0


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class SendVideoToDiscord:
    """An awaitable class that uploads a video file to one or more Discord webhooks.

    This action is designed to be driven by a `FireEventAction` container, whose
    background task calls it once per archived video chunk with the chunk's file
    path. If the file is within the configured size limit it is uploaded as a
    multipart attachment. If it is larger, or a webhook answers HTTP 413 (its
    real limit may be lower than configured), a short substitute text message
    with the file name and size is sent instead. The webhook URLs are secrets
    and are never logged (only the host is).

    Attributes:
        webhooks (list[str]): A list of Discord webhook URLs to upload to.
        upload_timeout_seconds (float): The total timeout for a single upload
            request, in seconds.
        file_limit_mb (float): Largest file uploaded, in MiB (1024 * 1024 bytes).
        substitute_message_template (str): The message sent instead of an
            upload. It may contain ``{file_name}`` and ``{size_mb}`` placeholders.
    """

    webhooks: list[str]
    upload_timeout_seconds: float = 60.0
    file_limit_mb: float = 18.0
    substitute_message_template: str = field(
        default=(
            "Goat on fire! A video chunk ({file_name}, {size_mb:.1f} MiB) is too large to upload to Discord. "
            "It is saved on disk."
        )
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

    async def _post_file(self, session: ClientSession, url: str, data: bytes, filename: str) -> bool:
        """Uploads the video file to a single Discord webhook URL.

        Args:
            session: The `aiohttp.ClientSession` to use for the request.
            url: The specific webhook URL to upload to.
            data: The raw bytes of the video file.
            filename: The filename to present to Discord.

        Returns:
            True if uploaded, False if Discord rejected the file as too large (HTTP 413).

        Raises:
            aiohttp.ClientError: If another non-2xx status code is received.
        """
        form = FormData()
        form.add_field("file", data, filename=filename, content_type="video/mp4")
        async with session.post(url=url, data=form, timeout=ClientTimeout(total=self.upload_timeout_seconds)) as response:
            if response.status == HTTP_PAYLOAD_TOO_LARGE:
                return False
            response.raise_for_status()
        return True

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
            results: The list of results (True, False for a 413 already reported, or exceptions) from `asyncio.gather`.
            verb: A short description of the operation for the log message.
        """
        success_count = 0
        for url, result in zip(self.webhooks, results, strict=True):
            if isinstance(result, Exception):
                sanitized_host = urlparse(url).hostname or "invalid-host"
                # aiohttp errors embed the request URL (a secret), so only the type and status are logged.
                status = getattr(result, "status", None)
                detail = f" HTTP {status}" if status else ""
                logger.error(f"Failed to {verb} to webhook {sanitized_host}: [{result.__class__.__name__}]{detail}")
            elif result is True:
                success_count += 1
        logger.info(f"Video chunk {verb} complete. Successful: {success_count}/{len(self.webhooks)}.")

    def _substitute_content(self, file_path: str, file_size: int) -> str:
        """Builds the substitute message (file name and size only, no directory)."""
        return self.substitute_message_template.format(file_name=os.path.basename(file_path), size_mb=file_size / BYTES_PER_MIB)

    async def _upload_file(self, session: ClientSession, file_path: str, file_size: int) -> None:
        """Reads and uploads the file concurrently to all configured webhooks.

        A webhook that answers HTTP 413 gets the substitute message instead.
        """
        try:
            data = await asyncio.to_thread(self._read_file, file_path)
        except OSError as e:
            logger.error(f"Could not read video chunk {os.path.basename(file_path)}: [{e.__class__.__name__}] {e}")
            return
        filename = os.path.basename(file_path)
        logger.info(f"Uploading video chunk {filename} to {len(self.webhooks)} webhook(s)...")
        results = await asyncio.gather(
            *(self._post_file(session, url, data, filename) for url in self.webhooks), return_exceptions=True
        )
        rejected = [url for url, result in zip(self.webhooks, results, strict=True) if result is False]
        if rejected:
            logger.warning(
                f"Discord rejected video chunk {filename} ({file_size / BYTES_PER_MIB:.1f} MiB) with HTTP 413 "
                f"(configured limit {self.file_limit_mb:g} MiB); sending a substitute message."
            )
            await self._send_substitute(session, rejected, file_path, file_size)
        self._log_results(results, "upload")

    async def _send_substitute(self, session: ClientSession, urls: list[str], file_path: str, file_size: int) -> None:
        """Sends the substitute message to the given webhooks."""
        content = self._substitute_content(file_path, file_size)
        results = await asyncio.gather(*(self._post_json(session, url, content) for url in urls), return_exceptions=True)
        for url, result in zip(urls, results, strict=True):
            if isinstance(result, Exception):
                host = urlparse(url).hostname or "invalid-host"
                logger.error(f"Failed to send substitute message to webhook {host}: [{type(result).__name__}]")

    async def __call__(self, file_path: str) -> None:
        """Uploads the given video chunk to Discord, or sends a substitute message.

        Args:
            file_path: The absolute path of the saved video chunk to send.
        """
        try:
            file_size = await asyncio.to_thread(os.path.getsize, file_path)
        except OSError as e:
            logger.error(f"Could not stat video chunk {os.path.basename(file_path)}: [{e.__class__.__name__}] {e}")
            return

        async with ClientSession() as session:
            if file_size <= self.file_limit_mb * BYTES_PER_MIB:
                await self._upload_file(session, file_path, file_size)
            else:
                logger.warning(
                    f"Video chunk {os.path.basename(file_path)} ({file_size / BYTES_PER_MIB:.1f} MiB) exceeds the "
                    f"{self.file_limit_mb:g} MiB limit; not uploading, sending a substitute message."
                )
                await self._send_substitute(session, self.webhooks, file_path, file_size)
