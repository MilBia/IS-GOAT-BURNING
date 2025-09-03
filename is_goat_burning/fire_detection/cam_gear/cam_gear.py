import logging as log

import cv2
from vidgear.gears.helper import logger_handler

from is_goat_burning.fire_detection.cam_gear.base_cam_gear import BaseYTCamGear

logger = log.getLogger("YTCamGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class YTCamGear(BaseYTCamGear):
    def _process_frame(self, frame):
        # apply colorspace to frames if valid
        if self.color_space is not None:
            color_frame = None
            try:
                if isinstance(self.color_space, int):
                    color_frame = cv2.cvtColor(frame, self.color_space)
                else:
                    raise ValueError(f"Global color_space parameter value `{self.color_space}` is not a valid!")
            except Exception as e:
                # Catch if any error occurred
                self.color_space = None
                if self._CamGear__logging:
                    logger.exception(str(e))
                    logger.warning("Input colorspace is not a valid colorspace!")
            if color_frame is not None:
                return color_frame
            return frame
        return frame
