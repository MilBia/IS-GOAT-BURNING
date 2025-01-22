from collections.abc import AsyncGenerator

import cv2
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


async def frame_gen_cl(stream: YTCamGear):
    async for frame in stream.read():
        if frame is None:
            break
        yield cv2.UMat(frame)


async def frame_gen_with_iterator_cl(stream: YTCamGear, iterator) -> AsyncGenerator[np.ndarray, np.ndarray]:
    async for frame in stream.read():
        if frame is None:
            break
        if await anext(iterator):
            yield cv2.UMat(frame)
