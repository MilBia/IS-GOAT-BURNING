import asyncio
import logging as log

from vidgear.gears import CamGear
from vidgear.gears.helper import logger_handler

from config import settings
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
        self.video_saver = AsyncVideoChunkSaver(
            fps=self.framerate,
            enabled=settings.video.save_video_chunks,
            output_dir=settings.video.video_output_directory,
            chunk_length_seconds=settings.video.video_chunk_length_seconds,
            max_chunks=settings.video.max_video_chunks,
            chunks_to_keep_after_fire=settings.video.chunks_to_keep_after_fire,
        )
        self.video_saver.start()

    def __init__(self, *args, **kwargs):
        self._pre__init__(*args, **kwargs)
        self.signal_handler = SignalHandler()
        super().__init__(*args, **kwargs)
        self.framerate = self.ytv_metadata.get("fps", 30)
        self.frame_wait_time = 1 / self.framerate - 0.01
        self._post__init__(*args, **kwargs)

    def _process_frame(self, frame):
        raise NotImplementedError

    async def read(self):
        loop = asyncio.get_running_loop()
        try:
            while True:
                self._CamGear__stream_read.clear()
                # Read the next frame in executor
                (grabbed, frame) = await loop.run_in_executor(None, self.stream.read)
                self._CamGear__stream_read.set()

                # check for valid frame if received
                if not grabbed:
                    break

                # Put the frame into the queue (this is non-blocking)
                self.video_saver(frame)

                processed_frame = await loop.run_in_executor(None, self._process_frame, frame)
                yield processed_frame
        finally:
            await self.video_saver.stop()
