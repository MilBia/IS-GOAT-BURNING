from collections.abc import Callable
from collections.abc import Generator

import numpy as np

from fire_detection.async_frame_generator import frame_gen
from fire_detection.async_frame_generator import frame_gen_with_iterator
from setting import CUDA
from setting import OPEN_CL

if OPEN_CL:
    from fire_detection.base_fire_detection.utils import _data_preparation
    from fire_detection.base_fire_detection.utils import _detect_fire
    from fire_detection.cam_gear.cam_gear_opencl import YTCamGear
elif CUDA:
    from fire_detection.base_fire_detection.utils import _cuda_data_preparation as _data_preparation
    from fire_detection.base_fire_detection.utils import _cuda_detect_fire as _detect_fire
    from fire_detection.cam_gear.cam_gear_cuda import YTCamGear
else:
    from fire_detection.base_fire_detection.utils import _data_preparation
    from fire_detection.base_fire_detection.utils import _detect_fire
    from fire_detection.cam_gear.cam_gear import YTCamGear


async def _detect_loop(
    stream: YTCamGear,
    margin: int,
    lower: np.ndarray,
    upper: np.ndarray,
    on_fire_action: Callable,
) -> None:
    """
    Main loop with reading frames from stream and detecting fire on them.

    :param stream: stream to fource
    :param margin: minimal number of pixels assumed as fire, needed to flag frame as containing fire
    :param lower: lower border of HSV values for fire color
    :param upper: upper border of HSV values for fire color
    :param on_fire_action: action to perform on case of fire detection
    """
    args = _data_preparation(lower, upper)
    async for frame in frame_gen(stream):
        fire = await _detect_fire(frame, margin, *args)

        if fire:
            await on_fire_action()


async def _detect_loop_with_frequency(
    stream: YTCamGear,
    margin: int,
    lower: np.ndarray,
    upper: np.ndarray,
    on_fire_action: Callable,
    iterator: Generator,
) -> None:
    """
    Main loop with reading frames from stream and detecting fire on them.

    :param stream: stream to fource
    :param margin: minimal number of pixels assumed as fire, needed to flag frame as containing fire
    :param lower: lower border of HSV values for fire color
    :param upper: upper border of HSV values for fire color
    :param on_fire_action: action to perform on case of fire detection
    :param iterator: generating which frame is destine to frame
    """
    args = _data_preparation(lower, upper)
    async for frame in frame_gen_with_iterator(stream, iterator):
        fire = await _detect_fire(frame, margin, *args)

        if fire:
            await on_fire_action()
