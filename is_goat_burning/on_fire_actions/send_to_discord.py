"""Provides a class to send Discord webhook notifications."""

import asyncio
from dataclasses import dataclass
from dataclasses import field
from urllib.parse import urlparse

from aiohttp import ClientResponse
from aiohttp import ClientSession

from is_goat_burning.logger import get_logger

logger = get_logger("DiscordSender")


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class SendToDiscord:
    """An awaitable class that sends a message to one or more Discord webhooks.

    This class uses `aiohttp` to send messages asynchronously and concurrently
    to a list of webhook URLs. It handles exceptions for each request
    individually.

    Attributes:
        webhooks (list[str]): A list of Discord webhook URLs to send messages to.
        message (str): The content of the message to be sent.
        data (dict[str, str]): The JSON payload sent to the webhook.
    """

    webhooks: list[str]
    message: str
    data: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Prepares the data payload after the instance is initialized."""
        self.data = {
            "content": self.message[:2000],  # Discord has a 2000 character limit
        }

    async def send_to_webhook(self, session: ClientSession, url: str) -> ClientResponse:
        """Sends the prepared message to a single Discord webhook URL.

        Args:
            session: The `aiohttp.ClientSession` to use for the request.
            url: The specific webhook URL to send the message to.

        Returns:
            The `aiohttp.ClientResponse` object from the request.

        Raises:
            aiohttp.ClientError: If a non-2xx status code is received.
        """
        async with session.post(
            url=url,
            json=self.data,
            headers={"User-Agent": "Python/3", "Content-Type": "application/json"},
            timeout=3,
        ) as response:
            # Raise an exception for non-2xx status codes to catch them later
            response.raise_for_status()
            return response

    async def __call__(self) -> None:
        """Sends messages concurrently to all configured webhooks.

        This method creates an `aiohttp.ClientSession` and gathers tasks for each
        webhook. It logs successes and failures individually, ensuring that one
        failed request does not prevent others from being sent.
        """
        async with ClientSession() as session:
            logger.info(f"Sending message to {len(self.webhooks)} webhook(s)...")
            tasks = [self.send_to_webhook(session, url) for url in self.webhooks]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = 0
            for url, result in zip(self.webhooks, results, strict=True):
                if isinstance(result, Exception):
                    # This catches TimeoutError, ClientError, etc.
                    sanitized_host = urlparse(url).hostname or "invalid-host"
                    logger.error(f"Failed to send to webhook {sanitized_host}: [{result.__class__.__name__}] {result}")
                else:
                    success_count += 1

            logger.info(f"Message sending complete. Successful: {success_count}/{len(self.webhooks)}.")
