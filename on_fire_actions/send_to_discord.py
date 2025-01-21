from dataclasses import dataclass, field

import asyncio

from aiohttp import ClientSession
import logging as log

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
            return response

    async def __call__(self):
        async with ClientSession() as session:
            logger.info("Sending messages...")
            tasks = [self.send_to_webhook(session, url) for url in self.webhooks]
            await asyncio.gather(*tasks)
            logger.info("Sent message to hooks")
