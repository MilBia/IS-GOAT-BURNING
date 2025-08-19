from config import settings

if settings.open_cl:
    from fire_detection.cam_gear.cam_gear_opencl import YTCamGear
elif settings.cuda:
    from fire_detection.cam_gear.cam_gear_cuda import YTCamGear
else:
    from fire_detection.cam_gear.cam_gear import YTCamGear

__all__ = ["YTCamGear"]
