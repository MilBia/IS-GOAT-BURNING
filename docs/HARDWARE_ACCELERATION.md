# Hardware Acceleration Guide (`hybrid_edge`)

This guide covers the optional, fully offline fire verifier of the `hybrid_edge` detection strategy (#72, #76, #78):
how it works, how to configure it, and what to install for each kind of hardware.

Statements are marked **(measured)** when they come from the research in #73 and **(unverified)** when they follow from
reading the code or documentation only and have not been run on real hardware.

> **Status:** no trained verifier model is published yet (open items of #73). `hybrid_edge` needs a model file that you
> provide through `EDGE__MODEL_PATH`; this repository ships none and `scripts/download_model.sh` has no default URL.

## How the verifier works

1. The classic local detector (HSV colour mask AND motion mask) runs on every analysed frame as a cheap gatekeeper.
2. Only when it fires, the frame is passed to the edge verifier: an **image classifier** (fire / no-fire, not an object
   detector) executed locally with `cv2.dnn` or ONNX Runtime. No network access is needed.
3. The verifier confirms fire when `P(fire) >= EDGE__CONFIDENCE_THRESHOLD`. The existing debounce logic applies afterwards.

A missing or unset `EDGE__MODEL_PATH` (or a path that is not an existing file) is a hard startup error: the `Settings`
validator raises at settings load and the application does not start. If the model exists but cannot be loaded (corrupt or
invalid model, `EDGE__INPUT_SIZE` mismatch, requested accelerator unavailable, `onnxruntime` missing, unexpected errors),
the detector factory logs a **CRITICAL** message and falls back to the classic local detector (honouring the CUDA/OpenCL
settings): fires are still reported, but **unverified**. At runtime, inference errors are treated as "no fire", logged at ERROR and escalate to
CRITICAL after 5 consecutive failures.

The loaded verifier is reused across stream reconnects: the model is read and warmed up once, not on every reconnect.
It is rebuilt only when the model file is replaced (mtime/size change) or an `EDGE__*` setting changes. A failed load is
never cached, so the next reconnect retries it (and logs the CRITICAL fallback again).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DETECTION_STRATEGY` | `classic` | Set to `hybrid_edge`. |
| `EDGE__MODEL_PATH` | none (required) | Path to the `.onnx` classifier. Must exist when `hybrid_edge` is selected. |
| `EDGE__ACCELERATOR` | `auto` | `auto`, `cpu`, `cuda` or `opencl`. See below. |
| `EDGE__CONFIDENCE_THRESHOLD` | `0.5` | Minimum `P(fire)` (0 to 1) that confirms a fire. |
| `EDGE__INPUT_SIZE` | `224` | Side of the square model input; must match the model. |

With `auto`, ONNX Runtime CUDA is tried when `onnxruntime` offers `CUDAExecutionProvider`; if CUDA is not the active
provider after all, the detector falls back to the `cv2.dnn` CPU engine with a WARNING. Without ONNX Runtime CUDA, `auto`
uses `cv2.dnn` on the CPU. OpenCL is never auto-selected. An explicit `cuda` that cannot be activated raises
`EdgeModelError` (and therefore the factory fallback described above), not a silent downgrade.

### Model contract

The model is a classifier exported to ONNX with **opset 12** (so `cv2.dnn` can read it). Preprocessing, as implemented
in `is_goat_burning/fire_detection/edge_detector.py`:

- BGR frame converted to RGB, squash-resized (no crop, aspect ratio not preserved) to `EDGE__INPUT_SIZE` x `EDGE__INPUT_SIZE`;
- scaled to 0-1, then ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`; NCHW `float32`, batch 1;
- output: two logits, `P(fire) = softmax(logits)[1]` (a single logit is treated as sigmoid).

The recommended architecture (MobileNetV3-Small at 224) and the training/export/evaluation commands are in
[`tools/edge_model/README.md`](../tools/edge_model/README.md); the benchmark harness is described in
[`tools/edge_model/README_bench.md`](../tools/edge_model/README_bench.md).

```bash
PYTHONPATH=. python -m tools.edge_model.train --manifest manifest.csv --arch mobilenetv3_small_100 --size 224 --out runs/mnv3s_224
PYTHONPATH=. python -m tools.edge_model.eval_onnx --manifest manifest.csv --model model.onnx
```

Fetch a published model with the generic downloader (HTTPS only, SHA-256 verified, atomic write, refuses to overwrite
without `--force`; you supply the URL and checksum):

```bash
scripts/download_model.sh https://example.com/fire_classifier.onnx <sha256> models/fire_classifier.onnx
```

## Per-accelerator setup

Note: the pip `opencv-python` wheel has **no Inference Engine (OpenVINO) backend and no CUDA support**; `cv2.dnn` there
runs on the CPU (and, in principle, OpenCL, see below). Intel NCS2/Myriad is **not supported** (dropped from
`Accelerator`, #102). OpenCV is not built from source for this feature.

### CPU (`EDGE__ACCELERATOR=cpu`, default via `auto`)

Uses `cv2.dnn`, which is already a project dependency: nothing extra to install. On a Raspberry Pi 4 it was as fast as or
faster than ONNX Runtime with 4 threads **(measured)**.

### NVIDIA CUDA (`EDGE__ACCELERATOR=cuda`)

Uses ONNX Runtime with the CUDA execution provider (not `cv2.dnn`, whose pip build has no CUDA).

Bare-metal install:

1. Install the optional extra: `pip install -e .[edge-gpu]` (pins `onnxruntime-gpu==1.32.0`; lock file
   `requirements-edge-gpu.txt`). Do not install `onnxruntime` and `onnxruntime-gpu` together.
2. **Constraint:** `onnxruntime-gpu` 1.32.0 needs the **CUDA 13** userspace libraries (`libcudart.so.13`,
   `libcublas.so.13`, `libcurand.so.10`, plus cuDNN) and an NVIDIA driver >= 580. They can come from the system or from
   pip (`onnxruntime-gpu[cuda,cudnn]`). This was taken from the package requirements and not run here **(unverified)**.

**Docker:** this is **not supported yet**. The repository's `gpu` image is based on `nvidia/cuda:12.4.1`, whose CUDA 12
libraries are not compatible with `onnxruntime-gpu` 1.32.0, so `EDGE__ACCELERATOR=cuda` fails there (CRITICAL log and
classic-detector fallback). Open question / planned: the `gpu` image already builds OpenCV with CUDA DNN
(`OPENCV_DNN_CUDA=ON`), so mapping `cuda` to the `cv2.dnn` CUDA backend inside that image is the likely follow-up. It is
not implemented and unverified. The NVIDIA Container Toolkit and `--gpus all` are needed for any future GPU container.

### Intel / AMD integrated GPU (`EDGE__ACCELERATOR=opencl`, experimental)

- Docker: `docker build --target opencl ...`, run with `--device /dev/dri:/dev/dri` and, if the device node is not
  accessible, `--group-add "$(getent group render | cut -d: -f3)"`. `/dev/dri` is never baked into the image.
- The container needs an OpenCL ICD: the `opencl` stage already installs `ocl-icd-libopencl1`, `clinfo`,
  `mesa-opencl-icd` and (amd64) `intel-opencl-icd`. `opencl-headers` is only needed to compile OpenCL code, so it is not
  added to the runtime image.
- **Honest caveat (measured, #73):** `cv2.dnn` OpenCL is unreliable on OpenCV 5.0.0. The new graph engine silently ignores
  `DNN_TARGET_OPENCL*` and runs on the CPU; the classic engine fails to compile a kernel, falls back to the CPU, can spend
  minutes auto-tuning, and FP16 was about 70x slower. Expect **no speed-up and possibly silent CPU execution**; the
  application logs a WARNING when this accelerator is selected. Prefer `cpu` unless `tools.check_acceleration` shows a
  real gain.
- The fastest backend measured on an Intel iGPU was OpenVINO GPU, which is **not integrated** (possible future optional
  work). The pip OpenCV wheel cannot use OpenVINO either.

## Measured latency (#73, random weights, batch 1, FP32)

Raspberry Pi 4 (4 cores, `opencv-python-headless` 5.0.0, `onnxruntime` 1.32.0), median of 40 runs:

| Model | cv2.dnn 4 thr | ORT 4 thr | cv2.dnn 1 thr | ORT 1 thr |
|---|---:|---:|---:|---:|
| MobileNetV3-Small 224 | 17.7 ms | 20.2 ms | 43.5 ms | 41.7 ms |
| MobileNetV3-Small 320 | 36.3 ms | 39.2 ms | 84.9 ms | 81.0 ms |
| MobileNetV2 224 | 45.2 ms | 53.8 ms | 109 ms | 118 ms |
| EfficientNet-B0 224 | 90.0 ms | 112.7 ms | 274 ms | 250 ms |

Intel Alder Lake-N UHD iGPU host (Docker, 224x224):

| Backend | MobileNetV3-Small | EfficientNet-B0 |
|---|---:|---:|
| ORT CPU 1 thr | 4.0 ms | 32.9 ms |
| cv2.dnn CPU (new engine) 1 thr | 7.7 ms | 47.1 ms |
| OpenVINO GPU (not integrated) | 3.8 ms | 8.1 ms |
| cv2.dnn OpenCL (classic engine) | 14.6 ms (no gain) | not run |
| cv2.dnn OpenCL FP16 (classic engine) | 275 ms (broken) | not run |

These are latency-only numbers with random weights. Accuracy on the real stream is unverified (see #73 open items). No
CUDA latency has been measured.

## Checking your hardware

```bash
PYTHONPATH=. python -m tools.check_acceleration                                      # probes only
PYTHONPATH=. python -m tools.check_acceleration --model models/fire.onnx --size 224  # + latency per accelerator
```

Inside a container: `tools/` is copied only into the `*-test` stages, so use a `*-test` image or mount the directory
(`-v "$PWD/tools:/app/tools:ro"`) and add `--device /dev/dri:/dev/dri` or `--gpus all` as appropriate, then run
`python3 -m tools.check_acceleration`.

The tool prints the OpenCV version, the DNN backend/target pairs OpenCV reports, OpenCL and `cv2.cuda` status, ONNX
Runtime providers, what `EDGE__ACCELERATOR=auto` resolves to and, with `--model`, `ok`/`error` plus the median latency
of 20 inferences (preprocessing included: 1080p frame resized and normalised to the input size, so not directly comparable to the model-only #73 numbers) for `auto`, `cpu`, `cuda` and `opencl`. It always exits with 0 (2 for invalid arguments). Treat an
`opencl: ok` result with a latency equal to `cpu` as "running on the CPU".

## Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| App refuses to start, `EDGE__MODEL_PATH` error | Path unset or not an existing file: hard startup error from the `Settings` validator. |
| CRITICAL log at startup, classic detector used | Model exists but cannot be loaded (invalid model, wrong `EDGE__INPUT_SIZE`, accelerator or `onnxruntime` unavailable). Read the error line above it; run `tools.check_acceleration --model ...`. |
| `EDGE__ACCELERATOR=cuda requires onnxruntime-gpu ...` | Extra not installed (`pip install -e .[edge-gpu]`), or running in the Docker `gpu` image, which is not supported for this (see above). |
| `CUDA is not the active onnxruntime provider` | CUDA 13 libraries / driver >= 580 missing (`onnxruntime-gpu` 1.32.0 needs them). |
| `Model ... expects a NxN input but EDGE__INPUT_SIZE=...` | Set `EDGE__INPUT_SIZE` to the model's input side. |
| `Test inference ... failed` | Not a 1/2-output classifier, or exported with an opset `cv2.dnn` cannot read (use opset 12). |
| `opencl requested but OpenCV reports no usable OpenCL runtime` | No ICD/device in the container: pass `--device /dev/dri`, add the `render` group, check `clinfo`. |
| OpenCL selected but not faster | Expected, see the caveat above; use `cpu`. |
| Repeated CRITICAL "inference failed N times" | Persistent backend errors; verification is effectively off until it recovers. |
