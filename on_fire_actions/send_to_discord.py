import asyncio
from dataclasses import dataclass
from dataclasses import field
import logging as log

from aiohttp import ClientSession
from vidgear.gears.helper import logger_handler

logger = log.getLogger("DiscordSender")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


@dataclass(init=True, repr=False, eq=False, order=False, kw_only=True, slots=True)
class SendToDiscord:
    webhooks: list[str]
    message: str
    data: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.data = {
            "content": self.message[:2000],
        }

    async def send_to_webhook(self, session, url):
        async with session.post(
            url=url,
            json=self.data,
            headers={"User-Agent": "Python/3", "Content-Type": "application/json"},
            timeout=3,
        ) as response:
            # Raise an exception for non-2xx status codes to catch them later
            response.raise_for_status()
            return response

    async def __call__(self):
        """
        Sends messages concurrently and handles exceptions for each request
        individually, ensuring that one failed request does not stop others.
        """
        async with ClientSession() as session:
            logger.info(f"Sending message to {len(self.webhooks)} webhook(s)...")
            tasks = [self.send_to_webhook(session, url) for url in self.webhooks]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = 0
            for url, result in zip(self.webhooks, results, strict=False):
                if isinstance(result, Exception):
                    # This catches TimeoutError, ClientError, etc.
                    logger.error(f"Failed to send to webhook {url}: [{result.__class__.__name__}] {result}")
                else:
                    success_count += 1

            logger.info(f"Message sending complete. Successful: {success_count}/{len(self.webhooks)}.")
