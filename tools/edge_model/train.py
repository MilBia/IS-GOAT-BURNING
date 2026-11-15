"""Fine-tune a small fire / no-fire image classifier and export it to ONNX (issue #73).

Input is a manifest CSV with columns ``path,label,split,source`` (see README). Images are squashed (no crop) to
``size`` x ``size`` and cached as uint8 arrays so that CPU-only training is practical.

Example:
    python -m tools.edge_model.train --manifest manifest.csv --arch mobilenetv3_small_100 --size 224 --out runs/mnv3s_224
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import timm
import torch
from torch import nn

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LABELS = {"no_fire": 0, "fire": 1}


def load_manifest(path: Path, smoke_policy: str) -> list[dict]:
    """Read the manifest and apply the smoke-only policy.

    Args:
        path: CSV manifest.
        smoke_policy: ``exclude`` drops ``smoke_only`` rows, ``positive`` maps them to fire, ``negative`` to no-fire.

    Returns:
        Rows with a numeric ``y`` field.
    """
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            label = row["label"]
            if label == "smoke_only":
                if smoke_policy == "exclude":
                    continue
                label = "fire" if smoke_policy == "positive" else "no_fire"
            row["y"] = LABELS[label]
            rows.append(row)
    return rows


def load_cached(rows: list[dict], size: int, cache: Path) -> np.ndarray:
    """Decode, squash-resize and cache images as an ``N,H,W,3`` RGB uint8 array."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        arr = np.load(cache)
        if len(arr) == len(rows):
            return arr
    out = np.empty((len(rows), size, size, 3), np.uint8)
    for i, row in enumerate(rows):
        img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["path"])
        out[i] = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    np.save(cache, out)
    return out


def to_tensor(batch: np.ndarray) -> torch.Tensor:
    """Normalise an ``N,H,W,3`` uint8 batch to a float ``N,3,H,W`` tensor."""
    t = torch.from_numpy(batch).float().div_(255.0).permute(0, 3, 1, 2)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (t - mean) / std


def augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap augmentation: horizontal flip, brightness/contrast jitter, small random crop-and-resize."""
    out = batch.copy()
    n, size = len(out), out.shape[1]
    for i in range(n):
        img = out[i]
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            crop = int(size * rng.uniform(0.75, 1.0))
            y0 = rng.integers(0, size - crop + 1)
            x0 = rng.integers(0, size - crop + 1)
            img = cv2.resize(np.ascontiguousarray(img[y0 : y0 + crop, x0 : x0 + crop]), (size, size))
        alpha = rng.uniform(0.7, 1.3)
        beta = rng.uniform(-25, 25)
        out[i] = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return out


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, bs: int = 128) -> np.ndarray:
    """Return P(fire) for every image."""
    model.eval()
    probs = []
    for i in range(0, len(x), bs):
        logits = model(to_tensor(x[i : i + bs]))
        probs.append(torch.softmax(logits, 1)[:, 1].numpy())
    return np.concatenate(probs)


def main() -> None:
    """Train, pick the best epoch on validation and export ONNX."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--arch", default="mobilenetv3_small_100", help="timm architecture name")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke-policy", choices=("exclude", "positive", "negative"), default="exclude")
    ap.add_argument("--gavle-weight", type=float, default=1.0, help="Loss weight for Gavle (hard negative) rows")
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest, args.smoke_policy)
    x_all = load_cached(rows, args.size, args.cache_dir / f"{args.manifest.stem}_{args.smoke_policy}_{args.size}.npy")
    y_all = np.array([r["y"] for r in rows])
    split = np.array([r["split"] for r in rows])
    w_all = np.array([args.gavle_weight if r["source"] == "gavle" else 1.0 for r in rows], np.float32)
    tr, va = np.where(split == "train")[0], np.where(split == "val")[0]
    print(f"train={len(tr)} (fire={y_all[tr].sum()}) val={len(va)} (fire={y_all[va].sum()})", flush=True)

    model = timm.create_model(args.arch, pretrained=True, num_classes=2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    # Balance classes through loss weights (fire is the minority in D-Fire).
    pos_w = float((y_all[tr] == 0).sum() / max(1, (y_all[tr] == 1).sum()))
    class_w = torch.tensor([1.0, min(pos_w, 3.0)])
    crit = nn.CrossEntropyLoss(weight=class_w, reduction="none")

    best = (-1.0, 0)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        t0, loss_sum = time.time(), 0.0
        for b in range(len(tr) // args.batch):
            idx = np.sort(order[b * args.batch : (b + 1) * args.batch])
            xb = to_tensor(augment(x_all[idx], rng))
            yb = torch.from_numpy(y_all[idx])
            wb = torch.from_numpy(w_all[idx])
            loss = (crit(model(xb), yb) * wb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item()
        p = predict(model, x_all[va])
        pred = p >= 0.5
        yv = y_all[va].astype(bool)
        recall = float((pred & yv).sum() / max(1, yv.sum()))
        fpr = float((pred & ~yv).sum() / max(1, (~yv).sum()))
        score = recall - 2 * fpr
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={loss_sum / (len(tr) // args.batch):.4f} "
            f"val_recall={recall:.3f} val_fpr={fpr:.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if score > best[0]:
            best = (score, epoch + 1)
            torch.save(model.state_dict(), args.out / "best.pt")

    model.load_state_dict(torch.load(args.out / "best.pt"))
    np.save(args.out / "val_probs.npy", predict(model, x_all[va]))
    for name in ("test",):
        te = np.where(split == name)[0]
        if len(te):
            np.save(args.out / f"{name}_probs.npy", predict(model, x_all[te]))
    meta = {
        "arch": args.arch,
        "size": args.size,
        "best_epoch": best[1],
        "smoke_policy": args.smoke_policy,
        "mean": MEAN,
        "std": STD,
        "color_order": "RGB",
        "classes": ["no_fire", "fire"],
        "output": "logits[2]; P(fire)=softmax(logits)[1]",
        "resize": "squash to size x size, INTER_AREA, no crop",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    export_onnx(model, args.size, args.out / "model.onnx")
    print("exported", args.out / "model.onnx")


def export_onnx(model: nn.Module, size: int, path: Path) -> None:
    """Export with the legacy TorchScript exporter at opset 12 so cv2.dnn can read it."""
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
