"""Convert a classifier ONNX into OpenVINO IR variants for #152.

Variants (written to ``--out/<stem>/``):

- ``fp32``: IR with FP32 weights (``compress_to_fp16=False``).
- ``fp16``: IR with FP16-compressed weights (OpenVINO decompresses them to FP32 on CPU at load time).
- ``int8w``: ``nncf.compress_weights`` INT8 asymmetric weight-only compression, no data.
- ``int8q``: ``nncf.quantize`` full post-training quantization (``ModelType.TRANSFORMER``) calibrated on frames from the
  edge-model manifest (D-Fire + Gavle ``train`` split only). Every candidate file of the calibration pool is hashed
  (SHA-256 of the bytes) and checked against all frames of ``--eval-manifest``; any overlap aborts.

The calibration frames go through the project's ``edge_detector.preprocess`` (copied below). A JSON log with the variant
file hashes and calibration counts (no image paths) is written to ``--out/<stem>/convert.json``. Keep ``--out`` outside
the repository; IR files and images are never committed.

Example:
    python -m tools.openvino_eval.convert ~/data/zero-shot-151/onnx/p16_224_minimal.onnx --out ~/data/openvino-152/ir \
        --calib-manifest ~/data/edge-model/manifest.csv --eval-manifest ~/data/llm-verifier-eval/manifest_main.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np

_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
VARIANTS = ("fp32", "fp16", "int8w", "int8q")


def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """Copy of ``is_goat_burning.fire_detection.edge_detector.preprocess`` (importing it needs the app config)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    return np.ascontiguousarray((blob - _MEAN) / _STD, dtype=np.float32)


def sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibration_paths(calib_manifest: Path, eval_manifest: Path, gavle: int, dfire: int, seed: int) -> tuple[list[Path], dict]:
    """Pick ``gavle`` + ``dfire`` train frames disjoint (by content hash) from every eval frame."""
    rows = [r for r in csv.DictReader(calib_manifest.open()) if r["split"] == "train"]
    eval_rows = json.loads(eval_manifest.read_text())
    eval_hashes = {sha256(eval_manifest.parent / r["path"]) for r in eval_rows}
    pool_hashes = {r["path"]: sha256(Path(r["path"])) for r in rows}
    overlap = sum(h in eval_hashes for h in pool_hashes.values())
    if overlap:
        raise SystemExit(f"{overlap} calibration-pool files are byte-identical to eval frames; aborting")
    rng = random.Random(seed)
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    picked = rng.sample(by_src.get("gavle", []), gavle) + rng.sample(by_src.get("dfire", []), dfire)
    labels: dict[str, int] = {}
    for r in picked:
        key = f"{r['source']}/{r['label']}"
        labels[key] = labels.get(key, 0) + 1
    info = {
        "pool": "edge-model manifest, split=train",
        "pool_size_by_source": {k: len(v) for k, v in sorted(by_src.items())},
        "eval_frames_hashed": len(eval_hashes),
        "pool_files_hashed": len(pool_hashes),
        "byte_identical_overlap": overlap,
        "picked_by_source_label": dict(sorted(labels.items())),
        "seed": seed,
    }
    return [Path(r["path"]) for r in picked], info


def main() -> None:
    """Write the IR variants and the conversion log."""
    import nncf
    import openvino as ov

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib-manifest", type=Path, required=True)
    ap.add_argument("--eval-manifest", type=Path, required=True)
    ap.add_argument("--calib-gavle", type=int, default=100)
    ap.add_argument("--calib-dfire", type=int, default=200)
    ap.add_argument("--seed", type=int, default=152)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = ap.parse_args()

    out = args.out / args.onnx.stem
    out.mkdir(parents=True, exist_ok=True)
    log: dict = {"onnx": args.onnx.name, "onnx_sha256": sha256(args.onnx), "openvino": ov.__version__, "nncf": nncf.__version__}
    size = int(ov.convert_model(args.onnx).input(0).get_partial_shape()[2].get_length())
    log["input_size"] = size
    for variant in args.variants:
        model = ov.convert_model(args.onnx)
        if variant == "int8w":
            model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
        elif variant == "int8q":
            paths, log["calibration"] = calibration_paths(
                args.calib_manifest, args.eval_manifest, args.calib_gavle, args.calib_dfire, args.seed
            )

            def transform(p: Path) -> np.ndarray:
                frame = cv2.imread(str(p))
                if frame is None:
                    raise FileNotFoundError(p)
                return preprocess(frame, size)

            model = nncf.quantize(
                model,
                nncf.Dataset(paths, transform),
                model_type=nncf.ModelType.TRANSFORMER,
                subset_size=len(paths),
            )
        xml = out / f"{variant}.xml"
        ov.save_model(model, xml, compress_to_fp16=(variant == "fp16"))
        log.setdefault("files", {})[variant] = {
            "xml_sha256": sha256(xml),
            "bin_sha256": sha256(xml.with_suffix(".bin")),
            "bin_mb": round(xml.with_suffix(".bin").stat().st_size / 1e6, 1),
        }
        print(variant, log["files"][variant], flush=True)
    (out / "convert.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
