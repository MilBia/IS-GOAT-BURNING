"""Zero-shot SigLIP2 fire verifier research (#151): ONNX export with built-in text embeddings, eval and bench.

No telemetry: onnxruntime uploads usage events to Microsoft unless ORT_DISABLE_TELEMETRY is set before it is
imported, and the Hugging Face hub sends usage info unless HF_HUB_DISABLE_TELEMETRY is set. Both are forced here,
before any submodule imports those libraries.
"""

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
