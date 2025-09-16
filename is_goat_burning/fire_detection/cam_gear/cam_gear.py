"""The standard CPU-based implementation of YTCamGear."""

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
    """A `YTCamGear` variant for standard CPU-based processing."""

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a video frame on the CPU.

        Applies a color space conversion if one is specified in the parent
        class's `color_space` attribute.

        Args:
            frame: The raw input frame as a NumPy array.

        Returns:
            The processed frame as a NumPy array.
        """
        if self.color_space is not None:
            color_frame: np.ndarray | None = None
            try:
                if isinstance(self.color_space, int):
                    color_frame = cv2.cvtColor(frame, self.color_space)
                else:
                    raise ValueError(f"Global color_space parameter value `{self.color_space}` is not a valid!")
            except Exception as e:
                self.color_space = None
                if self._CamGear__logging:
                    logger.exception(str(e))
                    logger.warning("Input colorspace is not a valid colorspace!")
            if color_frame is not None:
                return color_frame
        return frame
