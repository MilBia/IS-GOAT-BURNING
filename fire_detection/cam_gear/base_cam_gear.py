import asyncio
import logging as log

import cv2
from vidgear.gears import CamGear
from vidgear.gears.helper import logger_handler

from fire_detection.signal_handler import SignalHandler
from stream_recording.save_stream_to_file import AsyncVideoChunkSaver

logger = log.getLogger("YTCamGear")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


class BaseYTCamGear(CamGear):
    video_saver: AsyncVideoChunkSaver

    def _pre__init__(self, *args, **kwargs):
        kwargs["THREADED_QUEUE_MODE"] = False

    def _post__init__(self, *args, **kwargs):
        self.video_saver = AsyncVideoChunkSaver(fps=self.framerate)
        self.video_saver.start()

    def __init__(self, *args, **kwargs):
        self._pre__init__(*args, **kwargs)
        self.signal_handler = SignalHandler()
        super().__init__(*args, **kwargs)
        self.framerate = self.ytv_metadata.get("fps", 30)
        self.frame_wait_time = 1 / self.framerate - 0.01
        self._post__init__(*args, **kwargs)

    async def read(self):
        while self.signal_handler.KEEP_PROCESSING:
            # stream not read yet
            self._CamGear__stream_read.clear()

            # otherwise, read the next frame from the stream
            (grabbed, frame) = self.stream.read()

            # Put the frame into the queue (this is non-blocking)
            self.video_saver(frame)

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
            await asyncio.sleep(self.frame_wait_time)
        else:
            await self.video_saver.stop()
