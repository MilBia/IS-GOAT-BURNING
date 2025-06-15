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
    def _pre__init__(self, *args, **kwargs):
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        logger.info(f"Number of CUDA-capable GPUs: {device_count}")
        if device_count > 0:
            logger.info("CUDA enabled for processing.")
        else:
            logger.error("No CUDA-capable GPUs found")
            raise RuntimeError("No CUDA-capable GPUs found")
        kwargs["THREADED_QUEUE_MODE"] = True

    async def read(self):
        while self.signal_handler.KEEP_PROCESSING:
            # stream not read yet
            self._CamGear__stream_read.clear()

            # otherwise, read the next frame from the stream
            (grabbed, frame) = self.stream.read()

            # Put the frame into the queue (this is non-blocking)
            self.video_saver(frame)

            src = cv2.cuda.GpuMat()
            src.upload(frame)

            # stream read completed
            self._CamGear__stream_read.set()

            # check for valid frame if received
            if not grabbed:
                break

            # apply colorspace to frames if valid
            if self.color_space is not None:
                color_frame = None
                try:
                    if isinstance(self.color_space, int):
                        color_frame = cv2.cuda.cvtColor(src, self.color_space)
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
                    yield src
            else:
                yield src
            await asyncio.sleep(self.frame_wait_time)
        else:
            await self.video_saver.stop()
