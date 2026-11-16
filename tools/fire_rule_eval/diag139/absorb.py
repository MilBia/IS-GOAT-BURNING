# ruff: noqa: B007, B023, E401, E501, F401, I001, UP031
"""Exploratory diagnostic of #139 (run before the pre-registration; see bgmodel.py)."""

import numpy as np, cv2

rng = np.random.default_rng(1)
H, W = 270, 480
base = (rng.random((H, W)) * 60 + 80).astype(np.float32)


def frame(sq):
    g = base + rng.normal(0, 2.5, (H, W)).astype(np.float32)
    if sq:
        g[100:140, 200:260] = 240
    return np.clip(g, 0, 255).astype(np.uint8)


def run(make, k, lr_fixed=None, hist=30, n=60):
    m = make()
    out = []
    for i in range(n):
        g = frame(i >= k)
        lr = -1 if (lr_fixed is None or i < 5) else 1.0 / hist
        fg = m.apply(g, learningRate=lr) > 127 if hasattr(m, "apply") else None
        out.append(fg[100:140, 200:260].mean() if i >= k else 0)
    out = np.array(out[k:])
    # checks until fg share < 0.5
    below = np.flatnonzero(out < 0.5)
    return int(below[0]) if len(below) else ">%d" % len(out)


def med(N, k, n=60):
    buf = []
    out = []
    for i in range(n):
        g = frame(i >= k)
        if len(buf) >= 3:
            md = np.median(np.stack(buf), axis=0)
            fg = np.abs(g.astype(np.int16) - md) > 25
            out.append(fg[100:140, 200:260].mean() if i >= k else 0)
        else:
            out.append(0)
        buf = (buf + [g])[-N:]
    out = np.array(out[k:])
    below = np.flatnonzero(out < 0.5)
    return int(below[0]) if len(below) else ">%d" % len(out)


for k in (6, 15, 40):
    print("square appears at check", k)
    for H_ in (10, 30, 120):
        print(
            f"  mog2 h{H_} auto {run(lambda: cv2.createBackgroundSubtractorMOG2(H_, 16, False), k, None, H_)}  fixed-after-5 {run(lambda: cv2.createBackgroundSubtractorMOG2(H_, 16, False), k, 1, H_)}   knn h{H_} auto {run(lambda: cv2.createBackgroundSubtractorKNN(H_, 400.0, False), k, None, H_)} fixed {run(lambda: cv2.createBackgroundSubtractorKNN(H_, 400.0, False), k, 1, H_)}"
        )
    print(
        "  mog2 h30 v40 auto",
        run(lambda: cv2.createBackgroundSubtractorMOG2(30, 40, False), k, None, 30),
        "fixed",
        run(lambda: cv2.createBackgroundSubtractorMOG2(30, 40, False), k, 1, 30),
    )
    print("  med5", med(5, k), "med9", med(9, k))
