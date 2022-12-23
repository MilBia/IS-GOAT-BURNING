import asyncio

from fire_detection import detect_fire
from on_fire_actions import OnceAction, send_email


async def main():
    on_fire_action = OnceAction(
        send_email,
        sender="sender",
        sender_password="sender_password",
        recipients=["recipient1", "recipient2"],
        subject="GOAT ON FIRE!",
        message="Dear friend... Its time... Its time to Fight Fire With Fire.",
        host="smtp.gmail.com",
        port=587,
    )

    # await detect_fire(
    #     "https://youtu.be/6jge6uzRl-k",
    #     fire_border=0.05,
    #     logging=True,
    #     video_output=True,
    #     on_fire_action=on_fire_action,
    #     checks_per_second=1,
    # )
    await detect_fire(
        "https://youtu.be/TqvguE5cKT0",
        fire_border=0.1,
        logging=True,
        video_output=True,
        on_fire_action=on_fire_action,
        checks_per_second=1,
    )

    # await detect_fire(
    #     "https://youtu.be/6jge6uzRl-k",
    #     fire_border=0.05,
    #     logging=True,
    #     video_output=False,
    #     on_fire_action=on_fire_action,
    #     checks_per_second=1,
    # )
    # await detect_fire(
    #     "https://youtu.be/TqvguE5cKT0",
    #     fire_border=0.1,
    #     logging=True,
    #     video_output=False,
    #     on_fire_action=on_fire_action,
    #     checks_per_second=1,
    # )


if __name__ == "__main__":
    asyncio.run(main())
