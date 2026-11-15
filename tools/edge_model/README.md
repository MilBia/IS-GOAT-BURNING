# Edge verifier model selection (#73)

Research tooling and the recommendation for the local verifier used by `hybrid_edge` (#72). **No model weights are
published or committed.** See `README_bench.md` for the benchmark harness and iGPU results.

## Recommendation

**Architecture: an image classifier (fire / no-fire), not a detector. Primary: MobileNetV3-Small at 224x224. If the
runtime must also support NCS2/Myriad (#101), use MobileNetV2 (ReLU6) instead.**

| Property | MobileNetV3-Small 224 | MobileNetV2 224 | EfficientNet-B0 224 |
|---|---|---|---|
| RPi4 cv2.dnn, 4 thr (median) | 17.7 ms | 45.2 ms | 90.0 ms |
| RPi4 cv2.dnn, 1 thr | 43.5 ms | 109 ms | 274 ms |
| RPi4 at 320x320, 4 thr | 36.3 ms | 94.8 ms | 175.8 ms |
| salon (UHD iGPU) OpenVINO GPU | 3.8 ms | n/a | 8.1 ms |
| Operators (opset 12) | Conv, HardSigmoid, Mul, Relu, ... | Conv, Clip, ... | many shape ops |
| NCS2 (#101) | does not compile (HardSigmoid) | works | works |

- Every candidate is far below the issue's budget (< 500 ms on RPi4), so the choice is about operators and accuracy,
  not speed. `cv2.dnn` is the default backend on CPU: it is as fast as or faster than ONNX Runtime with 4 threads
  on the RPi4 (ORT is faster on a 1-thread MobileNetV3-Small run and on x86 for some models).
- Use OpenVINO GPU on Intel iGPUs. Do **not** rely on `cv2.dnn` OpenCL with OpenCV 5.0.0 (kernel build failure, FP16
  ~70x slower).
- YOLOX-Nano remains the fallback only if the classifier fails the acceptance metrics on real data.
- Numbers use random weights (latency only). RPi4 was measured with `bench.py` on 2026-11-15 and was ~10-20% slower than
  the first measurement in the issue (14.8 ms for MobileNetV3-Small), which is run-to-run variance on the same board.

## Pretrained models: not usable as-is

Checked 2026-11-15. All Apache-2.0 per their model cards, but trained on wildfire/smoke photos, so they do not
separate flames from the warm, flickering scenes the classic detector false-alarms on. Threshold picked on D-Fire val for
recall ~0.95, evaluated on the held-out D-Fire test split and on 68 test frames from two held-out Gavle days:

| Model | D-Fire recall | D-Fire FPR | Gavle false alarms |
|---|---:|---:|---:|
| `pyronear/mobilenet_v3_small` | 0.948 | 0.511 | 40 / 68 |
| `pyronear/resnet18` | 0.953 | 0.311 | 53 / 68 |

`pyronear/mobilenet_v3_large` and `pyronear/rexnet1_5x` were not evaluated. Their training data is described as
"soon to be shared", so provenance is unconfirmed. The ONNX files have a fixed batch size of 1.

## One fine-tuning experiment (not released)

A MobileNetV3-Small fine-tuned from timm ImageNet weights on D-Fire (fire vs. no-fire) plus real Gavle negatives met the
targets on D-Fire (test recall 0.952, FPR 0.004; 0/68 Gavle test frames). **This is weak evidence:** the Gavle test set is
~2 minutes of footage from one time of day, D-Fire splits are random frames of the same videos (near-duplicates leak
between train/val/test), and there is no real-stream fire footage. The weights are not published, so a model must be
trained (and re-evaluated on a larger real-stream set) before the first release asset exists.

## Data and licences

| Artifact | Licence | Decision |
|---|---|---|
| D-Fire | CC0 1.0 for the compilation | Use `AoF*` images. Exclude `PublicDataset*` (third-party terms, FESB asks for credit). Credit D-Fire. |
| D-Fire smoke-only images | n/a | Excluded from training and evaluation (neither positive nor negative). |
| FASDD | CC BY-SA 4.0 | Skipped (ShareAlike on weights is unsettled). |
| Gavle frames | own recordings | Not redistributed. Split by whole day. |
| timm / torchvision ImageNet weights | code Apache-2.0 / BSD-3 | ImageNet data is formally non-commercial. Whether weights inherit this is unsettled: note in any model card. |

The D-Fire copy used is the HuggingFace mirror `badsaarow/d-fire` (the official OneDrive links need a browser). Its
card declares no licence and its train/test split was not verified against the original.

## Tools

All scripts expect a manifest CSV `path,label,split,source` (`label`: `fire` / `no_fire` / `smoke_only`; `split`:
`train` / `val` / `test`; `source`: `dfire` / `gavle`). Training needs `torch`, `torchvision`, `timm` and `onnx`, which are
**not** project dependencies: use a separate venv.

```bash
export PYTHONPATH=.
# Fine-tune + export (legacy exporter, opset 12 so cv2.dnn can read it)
python -m tools.edge_model.train --manifest manifest.csv --arch mobilenetv3_small_100 --size 224 --out runs/mnv3s_224
python -m tools.edge_model.evaluate --manifest manifest.csv --run runs/mnv3s_224
# Evaluate any ONNX classifier (single logit -> sigmoid, two outputs -> softmax[1])
python -m tools.edge_model.eval_onnx --manifest manifest.csv --model model.onnx
# Latency on cv2.dnn / ONNX Runtime / OpenVINO
python -m tools.edge_model.bench model.onnx --threads 1 4
```

Preprocessing contract that the runtime (#76) must follow: RGB, squash-resize (no crop) to the model input size,
scale to 0-1, ImageNet mean `(0.485, 0.456, 0.406)` / std `(0.229, 0.224, 0.225)`, output `logits[2]`,
`P(fire) = softmax(logits)[1]`. Pass `dynamo=False, opset_version=12` to `torch.onnx.export`.

## Open items before closing #73

- Train and publish a classifier (release asset + model card) and evaluate on a larger, real-stream held-out set
  including dusk/night. Gavle recordings so far are all ~13:00 false alarms.
- Measure FP/hour on a real reference recording. The current 68 frames cannot support it.
- Re-check OpenVINO GPU accuracy (FP16 by default) with real weights.
