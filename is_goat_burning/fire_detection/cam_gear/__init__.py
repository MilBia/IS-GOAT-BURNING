from is_goat_burning.config import settings

if settings.open_cl:
    from is_goat_burning.fire_detection.cam_gear.cam_gear_opencl import YTCamGear
elif settings.cuda:
    from is_goat_burning.fire_detection.cam_gear.cam_gear_cuda import YTCamGear
else:
    from is_goat_burning.fire_detection.cam_gear.cam_gear import YTCamGear

__all__ = ["YTCamGear"]
