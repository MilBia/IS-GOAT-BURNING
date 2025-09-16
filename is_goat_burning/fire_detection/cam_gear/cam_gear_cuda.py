"""A CUDA-accelerated implementation of YTCamGear for NVIDIA GPUs."""

import logging as log
from typing import Any

import cv2
import numpy as np
from vidgear.gears.helper import logger_handler

from is_goat_burning.fire_detection.cam_gear.base_cam_gear import BaseYTCamGear

logger = log.getLogger("YTCamGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class YTCamGear(BaseYTCamGear):
    """A `YTCamGear` variant that uses CUDA for frame processing."""

    def _pre__init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Checks for CUDA device availability before initialization.

        Raises:
            RuntimeError: If no CUDA-capable GPU is found.
        """
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        logger.info(f"Number of CUDA-capable GPUs: {device_count}")
        if device_count == 0:
            raise RuntimeError("No CUDA-capable GPUs found")
        logger.info("CUDA enabled for processing.")
        kwargs["THREADED_QUEUE_MODE"] = False

    def _process_frame(self, frame: np.ndarray) -> cv2.cuda.GpuMat:
        """Processes a video frame using CUDA operations.

        Uploads the raw NumPy frame to the GPU, then performs an optional
        color space conversion using CUDA-accelerated functions.

        Args:
            frame: The raw input frame as a NumPy array.

        Returns:
            The processed frame as a `cv2.cuda.GpuMat` object residing on the
            GPU.
        """
        src = cv2.cuda.GpuMat()
        src.upload(frame)

        if self.color_space is not None:
            color_frame: cv2.cuda.GpuMat | None = None
            try:
                if isinstance(self.color_space, int):
                    color_frame = cv2.cuda.cvtColor(src, self.color_space)
                else:
                    raise ValueError(f"Global color_space parameter value `{self.color_space}` is not a valid!")
            except Exception as e:
                self.color_space = None
                if self._CamGear__logging:
                    logger.exception(str(e))
                    logger.warning("Input colorspace is not a valid colorspace!")
            if color_frame is not None:
                return color_frame
        return src
