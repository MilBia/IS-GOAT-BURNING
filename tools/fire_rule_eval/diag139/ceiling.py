# ruff: noqa: B007, B023, E401, E501, F401, I001, UP031
"""Exploratory diagnostic of #139 (run before the pre-registration; see bgmodel.py)."""

import numpy as np, glob

for f in sorted(glob.glob("/exp/138/features/synth_goat__*.npz")):
    d = np.load(f)
    a = d["area"][0]
    frac = float(f.split("_")[-1][:-4])
    w = d["warm"] / a
    t = d["t"]
    print(f.split("__")[-1], "checks", len(t))
    print("  t   ", np.round(t[3:20]).astype(int).tolist())
    print("  warm%", np.round(w[3:20] * 100, 2).tolist())
    ub = np.minimum(1.0, frac / np.maximum(w, 1e-9))
    print("  bound frac/warm (<=1)", np.round(ub[3:20], 3).tolist(), "median over t>=6:", float(np.median(ub[6:])))
