from collections.abc import AsyncGenerator

import numpy as np

from fire_detection.cam_gear import YTCamGear


async def frame_gen(stream: YTCamGear):
    async for frame in stream.read():
        if frame is None:
            break
        yield frame


async def frame_gen_with_iterator(stream: YTCamGear, iterator) -> AsyncGenerator[np.ndarray, np.ndarray]:
    async for frame in stream.read():
        if frame is None:
            break
        if await anext(iterator):
            yield frame
