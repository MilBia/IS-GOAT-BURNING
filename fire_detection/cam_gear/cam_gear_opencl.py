import asyncio
import logging as log

import cv2
from vidgear.gears.helper import logger_handler

from fire_detection.cam_gear.base_cam_gear import BaseYTCamGear

logger = log.getLogger("YTCamGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class YTCamGear(BaseYTCamGear):
    async def read(self):
        loop = asyncio.get_running_loop()
        try:
            while True:
                # stream not read yet
                self._CamGear__stream_read.clear()

                # otherwise, read the next frame from the stream
                (grabbed, frame) = await loop.run_in_executor(None, self.stream.read)

                # stream read completed
                self._CamGear__stream_read.set()

                # check for valid frame if received
                if not grabbed:
                    break

                # Put the frame into the queue (this is non-blocking)
                self.video_saver(frame)

                frame = cv2.UMat(frame)

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
                        yield color_frame
                    else:
                        yield frame
                else:
                    yield frame
        finally:
            await self.video_saver.stop()
