import asyncio

from fire_detection import detect_fire


async def main():
    await detect_fire("https://youtu.be/6jge6uzRl-k", fire_border=0.05, logging=True, video_output=True)
    # await detect_fire("https://youtu.be/TqvguE5cKT0", fire_border=0.05, logging=True, video_output=True)

    # await detect_fire("https://youtu.be/6jge6uzRl-k", fire_border=0.05, logging=True, video_output=False)
    # await detect_fire("https://youtu.be/TqvguE5cKT0", fire_border=0.05, logging=True, video_output=False)


if __name__ == "__main__":
    asyncio.run(main())
