"""Selects the appropriate YTCamGear implementation based on settings.

This module dynamically imports one of the `YTCamGear` variants (standard,
OpenCL, or CUDA) based on the `settings.open_cl` and `settings.cuda` flags,
ensuring that the rest of the application can import `YTCamGear` without
needing to know which backend is in use.
"""

from is_goat_burning.config import settings

if settings.open_cl:
    from is_goat_burning.fire_detection.cam_gear.cam_gear_opencl import YTCamGear
elif settings.cuda:
    from is_goat_burning.fire_detection.cam_gear.cam_gear_cuda import YTCamGear
else:
    from is_goat_burning.fire_detection.cam_gear.cam_gear import YTCamGear

__all__ = ["YTCamGear"]
