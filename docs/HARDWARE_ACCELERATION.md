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
   detector) executed locally with `cv2.dnn` or OpenVINO. No network access is needed.
3. The verifier confirms fire when `P(fire) >= EDGE__CONFIDENCE_THRESHOLD`. The existing debounce logic applies afterwards.

A missing or unset `EDGE__MODEL_PATH` (or a path that is not an existing file) is a hard startup error: the `Settings`
validator raises at settings load and the application does not start. If the model exists but cannot be loaded (corrupt or
invalid model, `EDGE__INPUT_SIZE` mismatch, requested accelerator unavailable, `openvino` missing, unexpected errors),
the detector factory logs a **CRITICAL** message and falls back to the classic local detector (honouring the CUDA/OpenCL
settings) and applies `VERIFIER_FAILURE_POLICY`: fires are reported **unverified** (`alert_unverified`) or not at all
(`suppress`). At runtime, inference errors are treated as "no fire", logged at ERROR and escalate to
CRITICAL after 5 consecutive failures.

The loaded verifier is reused across stream reconnects: the model is read and warmed up once, not on every reconnect.
It is rebuilt when the model file is replaced (mtime/size change; for an OpenVINO IR only the `.xml` is checked, so replace or touch it together with the `.bin`), an `EDGE__*` setting changes, or its last inference failed (a possibly broken session is not reused). A failed load is
never cached, so the next reconnect retries it (and logs the CRITICAL fallback again).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DETECTION_STRATEGY` | `classic` | Set to `hybrid_edge`. |
| `EDGE__MODEL_PATH` | none (required) | Path to the `.onnx` classifier (with `openvino` / `openvino_gpu` also an OpenVINO IR `.xml`, its `.bin` next to it). Must exist when `hybrid_edge` is selected. |
| `EDGE__ACCELERATOR` | `auto` | `auto`, `cpu`, `cuda`, `opencl`, `openvino` (OpenVINO, CPU) or `openvino_gpu` (OpenVINO, Intel iGPU). See below. |
| `EDGE__CONFIDENCE_THRESHOLD` | `0.5` | Minimum `P(fire)` (0 to 1) that confirms a fire. Valid only for the model variant **and** device it was calibrated on (see OpenVINO below). |
| `EDGE__INPUT_SIZE` | `224` | Side of the square model input; must match the model. |

With `auto`, `cv2.dnn` CUDA is used when OpenCV offers the dnn CUDA target
(`DNN_TARGET_CUDA in cv2.dnn.getAvailableTargets(DNN_BACKEND_CUDA)`; empty without cuDNN / `OPENCV_DNN_CUDA`, a usable
device or driver); if the model then cannot be loaded or warmed up on CUDA, the detector falls back to `cv2.dnn` on the
CPU with a WARNING. Without that target (including the pip `opencv-python` wheel, which has no CUDA), `auto` uses
`cv2.dnn` on the CPU. OpenCL and OpenVINO are never auto-selected. An explicit `cuda` raises `EdgeModelError` (and
therefore the factory fallback described above) when OpenCV reports no dnn CUDA target or the model fails to load or
run; layers without a CUDA kernel still run on the CPU silently (check with `OPENCV_LOG_LEVEL=INFO`, see below).

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
`Accelerator`, #102). Only the Docker `gpu` / `opencl` images build OpenCV from source (CUDA / OpenCL).

### CPU (`EDGE__ACCELERATOR=cpu`, default via `auto`)

Uses `cv2.dnn`, which is already a project dependency: nothing extra to install. On a Raspberry Pi 4 it was as fast as or
faster than ONNX Runtime (no longer used, #211) with 4 threads **(measured)**.

### NVIDIA CUDA (`EDGE__ACCELERATOR=cuda`)

Uses `cv2.dnn` with `DNN_BACKEND_CUDA` / `DNN_TARGET_CUDA` (FP32). There is no extra Python package: what matters is an
OpenCV **built with CUDA and cuDNN** (`OPENCV_DNN_CUDA=ON`). The pip `opencv-python` wheels have no CUDA, so on them
`cuda` is reported unavailable. Before #211 this path used `onnxruntime-gpu`; ONNX Runtime has been removed from the
project (its Linux wheels ship telemetry, and `onnxruntime-gpu` 1.32 needed CUDA 13, which the `gpu` image does not have).

> **Test status:** the CUDA path is verified **only by unit tests with mocks** (device detection, backend/target
> selection, `auto` fallback, explicit `cuda` error); no NVIDIA hardware was available, so it has **never run on a real
> GPU** and no CUDA latency has been measured **(unverified)**. Reports from users with an NVIDIA GPU are welcome.

- **Docker `gpu` image** (OpenCV 5.1 built with CUDA 12.4 and cuDNN 9, #214): build with
  `docker build --target gpu --build-arg CUDA_ARCH=<compute capability, e.g. 8.6> -t goat:gpu .`, install the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on the host and run
  with `docker run --gpus all ...` plus `DETECTION_STRATEGY=hybrid_edge`, `EDGE__ACCELERATOR=cuda` (or `auto`) and the
  model mounted at `EDGE__MODEL_PATH`. Without `--gpus all` there is no driver in the container,
  OpenCV reports no dnn CUDA target, and `cuda` fails to load (CRITICAL + classic-detector fallback) while
  `auto` uses the CPU.
- **Bare metal:** OpenCV must be compiled with `WITH_CUDA=ON`, `WITH_CUDNN=ON`, `OPENCV_DNN_CUDA=ON` (see
  `scripts/build_opencv.sh`, used by the Dockerfile) and its Python bindings installed instead of the `opencv-python`
  wheel; an NVIDIA driver is required.
- OpenCV 5's new DNN engine runs each layer on CUDA only if it has a CUDA implementation; other layers run on the CPU
  without an error. Check with `OPENCV_LOG_LEVEL=INFO`: the engine logs `finalize() backend=5 target=6 useCUDA=<0|1>`
  and one `finalize op ... -> CUDA|CPU` line per layer (`useCUDA=0` and all layers `-> CPU` were seen in the `gpu`
  image without a GPU). Which operators of the #73 / #151 models run on CUDA is **unverified**.
- `tools.check_acceleration` (`cuda devices (cv2.cuda)` line, `--model` latency for `cuda`) and
  `tools/edge_model/bench.py` (`cv2.dnn-cuda` row) show whether CUDA is usable.

### Intel / AMD integrated GPU (`EDGE__ACCELERATOR=opencl`, experimental)

- Docker: `docker build --target opencl ...`, run with `--device /dev/dri:/dev/dri` and, if the device node is not
  accessible, `--group-add "$(getent group render | cut -d: -f3)"`. `/dev/dri` is never baked into the image.
- The container needs an OpenCL ICD: the `opencl` stage already installs `ocl-icd-libopencl1`, `clinfo`,
  `mesa-opencl-icd` (Mesa rusticl) and (amd64) `intel-opencl-icd`. `opencl-headers` is only needed to compile OpenCL
  code, so it is not added to the runtime image.
- The `opencl` image is based on **Ubuntu 26.04** (the `cpu` image stays on 22.04) and ships Ubuntu's Intel compute
  runtime, `intel-opencl-icd` **26.05.x** (with `libigc2` / `libigdfcl2` / `libigdgmm12`; not pinned, it follows the 26.04
  archive at build time), which supports Gen12+ iGPUs
  such as Alder Lake-N (N100). The 22.04 runtime (22.14) listed no OpenCL device on such iGPUs, so OpenCV silently ran on
  the CPU (#223). Gen8-Gen11 iGPUs need Ubuntu's separate `intel-opencl-icd-legacy` package, which the image does not
  install. Check the device with
  `python3 -c "import cv2; print(cv2.ocl.Device.getDefault().name())"` (empty output: no usable device).
- **Honest caveat (measured, #73):** `cv2.dnn` OpenCL is unreliable on OpenCV 5.0.0. The new graph engine silently ignores
  `DNN_TARGET_OPENCL*` and runs on the CPU; the classic engine fails to compile a kernel, falls back to the CPU, can spend
  minutes auto-tuning, and FP16 was about 70x slower. OpenCV 5.1 (all images, #214) removed the classic engine, so
  `DNN_TARGET_OPENCL*` now always runs on the CPU (from the 5.1 source, not measured). Expect **no speed-up and silent
  CPU execution**; the application logs a WARNING when this accelerator is selected. Prefer `cpu` unless `tools.check_acceleration` shows a
  real gain.
- The fastest backend measured on an Intel iGPU was OpenVINO GPU, available as `EDGE__ACCELERATOR=openvino_gpu` (see
  below). The pip OpenCV wheel cannot use OpenVINO.

### OpenVINO (`EDGE__ACCELERATOR=openvino` / `openvino_gpu`, #207)

Uses the OpenVINO runtime directly (not through `cv2.dnn`): `openvino` runs on the CPU, `openvino_gpu` on an Intel
integrated GPU. Never selected by `auto`.

- On top of the base install (`pip install .[cpu]`), install from the lock file without dependency resolution:
  `pip install --no-deps -r requirements-openvino.txt`
  (fully pinned, `openvino==2026.4.1`, about +180 MB). It is compiled with `--unsafe-package=openvino-telemetry`, so the
  telemetry package is **not installed**; openvino then uses its built-in no-op stub. `pip check` reports
  "openvino 2026.4.1 requires openvino-telemetry, which is not installed": expected. `pip install -e .[openvino]` also
  works but pulls `openvino-telemetry` back in (the in-code block below still neutralises it). Without `openvino`,
  selecting either value fails to load (CRITICAL + classic-detector fallback).
- `EDGE__MODEL_PATH` may be the `.onnx` file or an OpenVINO IR `.xml` (e.g. an INT8 model quantised with NNCF, #152).
  The input size is read from the model and checked against `EDGE__INPUT_SIZE` as for the other backends.
- Compiled with `PERFORMANCE_HINT=LATENCY`; on the CPU with one stream and OpenVINO's default thread count (one per
  physical core). Each device uses its default precision: the CPU's own (f32 on the N100; bf16 on CPUs with native
  BF16/AMX), **f16 on the iGPU** (whatever the IR weights).
- `openvino_gpu` needs the Intel compute runtime and access to `/dev/dri` (`render` group). If OpenVINO reports no GPU
  device, the load fails (fallback as above); it never silently runs on the CPU.
- **Threshold:** `EDGE__CONFIDENCE_THRESHOLD` belongs to one model variant on one device. INT8 quantisation and iGPU f16
  shift `P(fire)`: in #152 the INT8 PTQ model on the N100 iGPU needed 0.8469, while the threshold derived from the
  CPU FP32 model (0.8082) gave 2 false positives on the INT8 model. Recalibrate after changing the model or the device.
- Measured in #152 (N100, SigLIP2 `p16-224` adapter, INT8 PTQ): median 170 ms on the CPU (4 threads, RSS about 390
  MiB) and 85 ms on the iGPU, against 401 ms and about 890 MiB with `cv2.dnn` **(measured)**.
- **Telemetry is disabled by the app.** The `openvino` wheel depends on `openvino-telemetry`, which on
  `import openvino` would send a usage event with a persistent client id (`~/intel/openvino_ga_cid`) to Google
  Analytics from a `multiprocessing` child (on Python 3.14 a forkserver that stays resident, about 105 MiB RSS plus a
  17 MiB resource tracker). Two layers prevent this: the lock-file install above keeps `openvino-telemetry` off disk,
  and, in case it is installed anyway, before importing `openvino` `OpenVinoBackend` makes `openvino_telemetry` unimportable
  (`sys.modules` entry set to `None`); `openvino` then uses its own built-in no-op stub. Verified with a temporary
  `HOME` and no opt-out file: no network attempt, no child process, nothing written to `~/intel`, identical `P(fire)`
  **(measured)**. No opt-out step is needed. The repository's own OpenVINO benchmark (`tools/edge_model/bench.py`) applies the
  same block; only external OpenVINO tools such as `ovc` are not covered.
- Docker: the images do not ship OpenVINO or the Intel GPU compute runtime yet (out of scope of #207); use a
  bare-metal install.

## Measured latency (#73, random weights, batch 1, FP32)

Raspberry Pi 4 (4 cores, `opencv-python-headless` 5.0.0, `onnxruntime` 1.32.0), median of 40 runs. The ORT columns are
historical (ONNX Runtime was removed from the project in #211):

| Model | cv2.dnn 4 thr | ORT 4 thr | cv2.dnn 1 thr | ORT 1 thr |
|---|---:|---:|---:|---:|
| MobileNetV3-Small 224 | 17.7 ms | 20.2 ms | 43.5 ms | 41.7 ms |
| MobileNetV3-Small 320 | 36.3 ms | 39.2 ms | 84.9 ms | 81.0 ms |
| MobileNetV2 224 | 45.2 ms | 53.8 ms | 109 ms | 118 ms |
| EfficientNet-B0 224 | 90.0 ms | 112.7 ms | 274 ms | 250 ms |

Intel Alder Lake-N UHD iGPU host (Docker, 224x224; the ORT row is historical, #211):

| Backend | MobileNetV3-Small | EfficientNet-B0 |
|---|---:|---:|
| ORT CPU 1 thr | 4.0 ms | 32.9 ms |
| cv2.dnn CPU (new engine) 1 thr | 7.7 ms | 47.1 ms |
| OpenVINO GPU (not integrated) | 3.8 ms | 8.1 ms |
| cv2.dnn OpenCL (classic engine) | 14.6 ms (no gain) | not run |
| cv2.dnn OpenCL FP16 (classic engine) | 275 ms (broken) | not run |

These are latency-only numbers with random weights. Accuracy on the real stream is unverified (see #73 open items). No
CUDA latency has been measured (no NVIDIA hardware).

## Checking your hardware

```bash
PYTHONPATH=. python -m tools.check_acceleration                                      # probes only
PYTHONPATH=. python -m tools.check_acceleration --model models/fire.onnx --size 224  # + latency per accelerator
```

Inside a container: `tools/` is copied only into the `*-test` stages, so use a `*-test` image or mount the directory
(`-v "$PWD/tools:/app/tools:ro"`) and add `--device /dev/dri:/dev/dri` or `--gpus all` as appropriate, then run
`python3 -m tools.check_acceleration`.

The tool prints the OpenCV version, the DNN backend/target pairs OpenCV reports, OpenCL and `cv2.cuda` status, what `EDGE__ACCELERATOR=auto` resolves to and, with `--model`, `ok`/`error` plus the median latency
of 20 inferences (preprocessing included: 1080p frame resized and normalised to the input size, so not directly comparable to the model-only #73 numbers) for `auto`, `cpu`, `cuda`, `opencl`, `openvino` and `openvino_gpu`. It always exits with 0 (2 for invalid arguments). Treat an
`opencl: ok` result with a latency equal to `cpu` as "running on the CPU".

## Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| App refuses to start, `EDGE__MODEL_PATH` error | Path unset or not an existing file: hard startup error from the `Settings` validator. |
| CRITICAL log at startup, classic detector used | Model exists but cannot be loaded (invalid model, wrong `EDGE__INPUT_SIZE`, accelerator or `openvino` unavailable). Read the error line above it; run `tools.check_acceleration --model ...`. |
| `The 'openvino' package is required ...` | Install it: `pip install --no-deps -r requirements-openvino.txt`. |
| `openvino_gpu requested but OpenVINO reports no GPU device` | Intel compute runtime missing or no access to `/dev/dri` (add the user to the `render` group); the Docker images do not support it yet. |
| `EDGE__ACCELERATOR=cuda requested but OpenCV reports no cv2.dnn CUDA target` | OpenCV without CUDA (pip wheel: use the Docker `gpu` image or a CUDA build), OpenCV built without cuDNN / `OPENCV_DNN_CUDA`, container started without `--gpus all` / NVIDIA Container Toolkit, or no NVIDIA driver. |
| `EDGE__ACCELERATOR=cuda requested but unusable: ...` / WARNING `cv2.dnn CUDA could not be used` (`auto`) | OpenCV offers CUDA but the model failed to load or run on it (e.g. `CUDA_ARCH` not matching the GPU, out of memory). Read the quoted error. |
| `Model ... expects a NxN input but EDGE__INPUT_SIZE=...` | Set `EDGE__INPUT_SIZE` to the model's input side. Checked at startup on both backends from the input shape declared in the ONNX file (parsed directly for `cv2.dnn`); models with a dynamic H/W are not checked, so a wrong size there only surfaces as degraded accuracy. |
| `Test inference ... failed` | Not a 1/2-output classifier, or exported with an opset `cv2.dnn` cannot read (use opset 12). |
| `opencl requested but OpenCV reports no usable OpenCL runtime` | No ICD/device in the container: pass `--device /dev/dri`, add the `render` group, check `clinfo`. |
| OpenCL selected but not faster | Expected, see the caveat above; use `cpu`. |
| Repeated CRITICAL "inference failed N times" | Persistent backend errors; verification is effectively off until it recovers. |
