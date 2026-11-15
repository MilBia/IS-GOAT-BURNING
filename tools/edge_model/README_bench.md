# Edge model benchmark (#73)

`bench.py` measures median/p95 latency of ONNX classifiers on cv2.dnn (CPU, OpenCL, OpenCL FP16), onnxruntime CPU and
OpenVINO (CPU, GPU), plus the max abs output diff vs ORT CPU. Backends that are not importable are skipped with a note.

```bash
python -m tools.edge_model.bench model.onnx [model2.onnx ...] --runs 40 --warmup 5 --threads 1 4 [--cv2-engine classic|new]
```

`--cv2-engine`: OpenCV 5 has a "new graph engine" that silently ignores `DNN_TARGET_OPENCL*` (warning
`Targets are not supported by the new graph engine`), so those rows would really be CPU. Default `classic` forces the old engine.

## Running on an Intel iGPU host in Docker (tested: Alder Lake-N UHD, Ubuntu host)

`Dockerfile.bench` (in this directory) installs `intel-opencl-icd` (compute runtime), opencv-python-headless 5.x,
onnxruntime and openvino 2026.x. Pass the render node and the host `render` group (here gid 990; check `getent group render`).

```bash
mkdir -p ~/edge-bench-salon/{models,tools} && cp -r tools/edge_model tools/__init__.py ~/edge-bench-salon/tools/
cp mnv3s_224.onnx effb0_224.onnx ~/edge-bench-salon/models/
cd ~/edge-bench-salon
docker build -f tools/edge_model/Dockerfile.bench -t edge-bench-salon .
R="docker run --rm --device /dev/dri --group-add 990 -v $PWD:/work edge-bench-salon"
$R clinfo -l                                   # expect: Intel(R) UHD Graphics
$R python -c "import openvino; print(openvino.Core().available_devices)"   # ['CPU', 'GPU']
$R python -m tools.edge_model.bench models/mnv3s_224.onnx models/effb0_224.onnx --cv2-engine new  2>/dev/null
# classic engine for the OpenCL rows: very slow (minutes) and extremely noisy on stderr, so redirect it
$R python -m tools.edge_model.bench models/mnv3s_224.onnx --cv2-engine classic --runs 20 2>classic.err
```

## Results (salon, 4 cores, UHD iGPU, 224x224, ONNX opset 12 from goatpi, ms)

Run in the container with opencv 5.0.0, onnxruntime 1.32.0, openvino 2026.4.1. The ONNX files are the random-weight ones, so
outputs are ~1e-10 and the absolute diffs are only a sanity check, not an accuracy measure.

| backend | mnv3s_224 median / p95 | effb0_224 median / p95 |
|---|---:|---:|
| ort-cpu 1 thread | 4.03 / 4.22 | 32.9 / 33.6 |
| ort-cpu 4 threads | 5.6-7.5 / ~10 (noisy) | 14.9 / 15.5 |
| cv2.dnn CPU, new engine, 1 / 4 thr | 7.7 / 8.0 | 47.1 / 20.8 |
| cv2.dnn CPU, classic engine, 1 / 4 thr | 11.5 / 11.4 | n/a |
| cv2.dnn OpenCL (classic engine) | 14.6 (no gain over CPU) | n/a |
| cv2.dnn OpenCL FP16 (classic engine) | 275 (broken) | n/a |
| cv2.dnn OpenCL, new engine | falls back to CPU (targets ignored) | falls back to CPU |
| openvino CPU 1 / 4 thr | 5.8 / 4.0 | 40.1 / 18.0 |
| **openvino GPU (iGPU)** | **3.8 / 3.9** | **8.1 / 8.2** |

Findings:
- OpenVINO GPU is the only backend that really uses the iGPU well (EfficientNet-B0: 8 ms vs 15 ms best CPU).
  Max abs diff vs ORT CPU 3e-10 on outputs of magnitude 3e-10 (to be re-checked with real weights; the GPU plugin runs FP16 inference by default).
- cv2.dnn OpenCL with OpenCV 5.0.0 is not usable: the new engine ignores the target, the classic engine fails to
  compile the `dnn/activations` kernel (`std::pow` in OpenCL C, CL_BUILD_PROGRAM_FAILURE), falls back to CPU for those
  layers, takes minutes (kernel auto-tuning) and FP16 is ~70x slower.
- For a 1 fps-class detector, MobileNetV3-Small at ~4 ms is fine on any CPU backend; the iGPU only matters for larger models.
