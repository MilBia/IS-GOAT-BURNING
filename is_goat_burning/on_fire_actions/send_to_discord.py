"""Provides a class to send Discord webhook notifications."""

import asyncio
from dataclasses import dataclass
from dataclasses import field
import json
import math
from urllib.parse import urlparse

from aiohttp import ClientError
from aiohttp import ClientResponse
from aiohttp import ClientSession
from aiohttp import ClientTimeout
from aiohttp import FormData
import cv2
import numpy as np

from is_goat_burning.logger import get_logger

logger = get_logger("DiscordSender")

HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500
DISCORD_MESSAGE_LIMIT = 2000
FRAME_MAX_WIDTH = 1280
FRAME_JPEG_QUALITY = 85
UNVERIFIED_PREFIX = "\n\n[UNVERIFIED] "
UNVERIFIED_NOTE = "The fire verifier is unavailable; this alert comes from the local camera detector only."


def _host(url: str) -> str:
    """Returns only the host of a webhook URL (the full URL is a secret)."""
    return urlparse(url).hostname or "invalid-host"


def _encode_frame(frame: np.ndarray) -> bytes:
    """Downscales a frame to at most `FRAME_MAX_WIDTH` px wide and encodes it as JPEG (blocking; run in a thread).

    Raises:
        ValueError: If the frame is empty or cannot be encoded.
    """
    if frame.size == 0:
        raise ValueError("empty frame")
    height, width = frame.shape[:2]
    if width > FRAME_MAX_WIDTH:
        scale = FRAME_MAX_WIDTH / width
        frame = cv2.resize(frame, (FRAME_MAX_WIDTH, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class SendToDiscord:
    """An awaitable class that sends a message to one or more Discord webhooks.

    Messages are sent concurrently with `aiohttp`. Each webhook is retried with
    exponential backoff on connection errors, timeouts, HTTP 5xx and HTTP 429
    (honouring `Retry-After` / JSON `retry_after`, capped at `max_delay`).
    Other 4xx responses are permanent and not retried. A small explicit loop is
    used instead of `tenacity` (not a project dependency) because the 429 delay
    comes from the response and the logic is short.

    Attributes:
        webhooks (list[str]): A list of Discord webhook URLs to send messages to.
        message (str): The content of the message to be sent.
        data (dict[str, str]): The JSON payload sent to the webhook.
        timeout (float): Per-request timeout in seconds.
        max_attempts (int): Maximum number of attempts per webhook.
        base_delay (float): Delay before the first retry; doubles each retry.
        max_delay (float): Upper bound for any single wait between attempts.
        attach_frame (bool): If True, a frame passed to the call is attached to the message as a JPEG.
        unverified_note (str): Appended as ``[UNVERIFIED] <note>`` to an alert sent while the fire verifier was
            unavailable.
    """

    webhooks: list[str]
    message: str
    data: dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    attach_frame: bool = True
    unverified_note: str = UNVERIFIED_NOTE

    def __post_init__(self) -> None:
        """Prepares the data payload after the instance is initialized."""
        self.data = {
            "content": self.message[:DISCORD_MESSAGE_LIMIT],  # Discord has a 2000 character limit
        }

    def _payload(self, unverified: bool) -> dict[str, str]:
        """Returns the JSON payload; an unverified alert gets the note appended (kept within the limit)."""
        if not unverified:
            return self.data
        note = UNVERIFIED_PREFIX + self.unverified_note
        return {"content": self.message[: DISCORD_MESSAGE_LIMIT - len(note)] + note}

    async def _retry_after(self, response: ClientResponse) -> float | None:
        """Extracts the server-requested wait (seconds) from a 429 response.

        Returns None when absent or not a finite number (caller uses backoff).
        """
        candidates: list[object] = [response.headers.get("Retry-After")]
        if candidates[0] is None:
            try:
                candidates = [(await response.json()).get("retry_after")]
            except ClientError, ValueError, AttributeError:
                return None
        try:
            value = float(candidates[0])  # type: ignore[arg-type]
        except TypeError, ValueError:
            return None
        return value if math.isfinite(value) else None

    async def send_to_webhook(
        self, session: ClientSession, url: str, payload: dict[str, str] | None = None, image: bytes | None = None
    ) -> bool:
        """Sends the message to one webhook, retrying transient failures.

        Args:
            session: The `aiohttp.ClientSession` to use for the request.
            url: The webhook URL (never logged; only its host is).
            payload: The JSON payload; defaults to `data`.
            image: JPEG bytes to attach; the request is then multipart (`payload_json` + `files[0]`).

        Returns:
            True if the webhook accepted the message, False otherwise.
        """
        host = _host(url)
        for attempt in range(1, self.max_attempts + 1):
            delay = min(self.base_delay * 2 ** (attempt - 1), self.max_delay)
            try:
                body = self.data if payload is None else payload
                if image is None:
                    request: dict[str, object] = {
                        "json": body,
                        "headers": {"User-Agent": "Python/3", "Content-Type": "application/json"},
                    }
                else:
                    # aiohttp's FormData is single-use, so it is rebuilt for every attempt.
                    form = FormData()
                    form.add_field("payload_json", json.dumps(body), content_type="application/json")
                    form.add_field("files[0]", image, filename="frame.jpg", content_type="image/jpeg")
                    request = {"data": form, "headers": {"User-Agent": "Python/3"}}
                async with session.post(
                    url=url,
                    timeout=ClientTimeout(total=self.timeout),
                    **request,  # type: ignore[arg-type]
                ) as response:
                    status = response.status
                    if status < 400:  # noqa: PLR2004
                        return True
                    if status == HTTP_TOO_MANY_REQUESTS:
                        requested = await self._retry_after(response)
                        if requested is not None:
                            delay = min(max(requested, 0.0), self.max_delay)
                    elif status < HTTP_SERVER_ERROR:
                        logger.error(f"Webhook {host} rejected the message with HTTP {status}; not retrying.")
                        return False
                    reason = f"HTTP {status}"
            except (TimeoutError, ClientError) as exc:
                reason = exc.__class__.__name__
            if attempt == self.max_attempts:
                logger.error(f"Webhook {host} failed after {attempt} attempt(s): {reason}.")
                return False
            logger.warning(f"Webhook {host} attempt {attempt}/{self.max_attempts} failed ({reason}); retrying in {delay:.1f}s.")
            await asyncio.sleep(delay)
        return False

    async def __call__(self, *, unverified: bool = False, frame: np.ndarray | None = None) -> bool:
        """Sends messages concurrently to all configured webhooks.

        Args:
            unverified: If True, `unverified_note` is appended so the recipient can tell an alert
                sent while the fire verifier was unavailable from a verified one.
            frame: The frame that triggered the alert. Attached as a JPEG when `attach_frame` is set; if it
                cannot be encoded the text alert is sent without it.

        Returns:
            True if at least one webhook accepted the message, False if none did
            (so the caller can keep the alert armed for a later attempt).
        """
        payload = self._payload(unverified)
        image: bytes | None = None
        if self.attach_frame and frame is not None:
            try:
                image = await asyncio.to_thread(_encode_frame, frame)
            except Exception as exc:  # noqa: BLE001 - the image is optional; never block the alert
                logger.warning(f"Could not encode the alert frame ({type(exc).__name__}); sending the text only.")
        async with ClientSession(trust_env=True) as session:  # honours HTTPS_PROXY (egress proxy, #212)
            logger.info(f"Sending message to {len(self.webhooks)} webhook(s)...")
            results = await asyncio.gather(
                *(self.send_to_webhook(session, url, payload, image) for url in self.webhooks), return_exceptions=True
            )
        for url, result in zip(self.webhooks, results, strict=True):
            if isinstance(result, Exception):  # Unexpected error: isolate it so other webhooks still count.
                logger.error(f"Unexpected error sending to webhook {_host(url)}: [{result.__class__.__name__}]")
        success_count = sum(1 for result in results if result is True)
        logger.info(f"Message sending complete. Successful: {success_count}/{len(self.webhooks)}.")
        return success_count > 0
