"""An OpenCL-accelerated implementation of YTCamGear."""

import logging as log

import cv2
import numpy as np
from vidgear.gears.helper import logger_handler

from is_goat_burning.fire_detection.cam_gear.base_cam_gear import BaseYTCamGear

logger = log.getLogger("YTCamGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class YTCamGear(BaseYTCamGear):
    """A `YTCamGear` variant that uses OpenCL for frame processing."""

    def _process_frame(self, frame: np.ndarray) -> cv2.UMat:
        """Processes a video frame using OpenCL-aware operations.

        Converts the incoming NumPy array to a `cv2.UMat` object. Subsequent
        OpenCV operations (like `cvtColor`) can then be transparently
        accelerated by an available OpenCL device.

        Args:
            frame: The raw input frame as a NumPy array.

        Returns:
            The processed frame as a `cv2.UMat` object.
        """
        frame_umat = cv2.UMat(frame)

        if self.color_space is not None:
            color_frame: cv2.UMat | None = None
            try:
                if isinstance(self.color_space, int):
                    color_frame = cv2.cvtColor(frame_umat, self.color_space)
                else:
                    raise ValueError(f"Global color_space parameter value `{self.color_space}` is not a valid!")
            except Exception as e:
                self.color_space = None
                if self._CamGear__logging:
                    logger.exception(str(e))
                    logger.warning("Input colorspace is not a valid colorspace!")
            if color_frame is not None:
                return color_frame
        return frame_umat
