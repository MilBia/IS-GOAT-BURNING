"""Independent scene-cut labelling of the real fire clips (#138, POST HOC analysis; not a gl_* feature).

A check ``i`` is labelled a *cut* when the 3-D colour histogram (8 bins per channel, 480x270 frames) of check ``i`` differs
from that of check ``i - 1`` by a Bhattacharyya distance above ``CUT_DISTANCE``. The detector uses neither the grey
difference, the tiles nor the warm mask, so it does not share information with the gate it is used to audit. The labels
are checked by eye on montages (see the report) and the resulting list is committed BEFORE the first ``compare``.

Usage: ``python tools/fire_rule_eval/cuts.py MANIFEST OUT.json [MONTAGE_DIR]``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

from tools.fire_rule_eval.features import ExtractOptions
from tools.fire_rule_eval.features import iter_check_frames

CUT_DISTANCE = 0.3
BINS = 8


def histogram(frame: np.ndarray) -> np.ndarray:
    """Normalised 3-D colour histogram of a frame."""
    hist = cv2.calcHist([frame], [0, 1, 2], None, [BINS] * 3, [0, 256] * 3)
    return cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1).astype(np.float32)


def cut_distances(path: Path, montage: Path | None = None) -> list[float]:
    """Bhattacharyya distance between the colour histograms of consecutive checks (0 for the first)."""
    out: list[float] = []
    previous = None
    previous_frame = None
    for i, (_, frame) in enumerate(iter_check_frames(path, ExtractOptions(1.0, 0.0, None, 0.25))):
        hist = histogram(frame)
        distance = 0.0 if previous is None else float(cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA))
        out.append(distance)
        if montage is not None and distance > CUT_DISTANCE * 0.6:
            pair = np.hstack([cv2.resize(previous_frame, (240, 135)), cv2.resize(frame, (240, 135))])
            cv2.putText(
                pair, f"{path.stem[-18:]} i={i} d={distance:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )
            montage.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(montage / f"{path.stem[-30:]}_{i:03d}.jpg"), pair)
        previous, previous_frame = hist, frame
    return out


def main() -> int:
    """Writes ``{clip: {"distances": [...], "cuts": [check indices]}}`` for the real fire clips of a manifest."""
    manifest = json.loads(Path(sys.argv[1]).read_text())
    montage = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    result = {}
    for entry in manifest["clips"]:
        if not entry.get("group", "").startswith("commons_fire"):
            continue
        distances = cut_distances(Path(entry["path"]), montage)
        result[entry["name"]] = {
            "distances": [round(d, 3) for d in distances],
            "cuts": [i for i, d in enumerate(distances) if d > CUT_DISTANCE],
        }
        print(entry["name"], len(distances), "checks; cuts at", result[entry["name"]]["cuts"], flush=True)
    Path(sys.argv[2]).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
