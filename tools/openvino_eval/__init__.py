"""OpenVINO runtime research tools (#152).

No telemetry: onnxruntime (used for the reference scores) uploads usage events to Microsoft unless
ORT_DISABLE_TELEMETRY is set before it is imported; forced here, before any submodule imports it. OpenVINO/NNCF
telemetry is blocked in ``convert.py`` and ``bench.py``.
"""

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
