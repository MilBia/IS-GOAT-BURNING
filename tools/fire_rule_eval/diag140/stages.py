"""POST HOC: stage breakdown of the eigenbackground cost at 480x270 (where do the milliseconds go; is amortisation possible?).

Stages for ``eig`` with N = 9 previous frames, rank 1, 1 thread: conversion + shift, the window matvec ``X @ g`` (also the Gram
update), the small eigendecomposition, the background matvec ``X^T a``, the threshold mask, and the counting of the warm AND
foreground pixels. Real goat frames, decode excluded.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from tools.fire_rule_eval.features import ExtractOptions
from tools.fire_rule_eval.features import iter_check_frames
from tools.fire_rule_eval.lowrank import MOTION_THRESHOLD
from tools.fire_rule_eval.lowrank import LowRankModels

VIDEO = Path("/exp/data/goat/event_2025-12-27_13-00-33/goat-cam_2025-12-27_13-01-11.mp4")


def _ms(step: Callable[[], object], repeats: int = 200) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        step()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main() -> None:
    """Prints the median time of each stage."""
    cv2.setNumThreads(1)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in iter_check_frames(VIDEO, ExtractOptions(1.0, 0.0, 12, 0.25))]
    models = LowRankModels(("lr_eig1_n9",))
    for gray in grays:
        models.masks(gray)
    model = models._models["lr_eig1_n9"]
    window = model.window
    stack = window.view()
    gram = window.gram[: len(stack), : len(stack)]
    gray = grays[-1]
    flat = window.prepare(gray)
    products = stack @ flat
    centre = np.eye(len(stack)) - 1.0 / len(stack)
    a = np.full(len(stack), 1.0 / len(stack), dtype=np.float32)
    background = a @ stack
    warm = cv2.inRange(cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV), (0, 0, 0), (179, 255, 255))
    fg = np.where(np.abs(flat - background) > MOTION_THRESHOLD, 255, 0).astype(np.uint8).reshape(gray.shape)
    stages = {
        "convert + shift (astype, subtract reference)": lambda: window.prepare(gray),
        "window matvec X @ g (Gram row)": lambda: stack @ flat,
        "eigh of the 9x9 centred Gram matrix": lambda: np.linalg.eigh(centre @ gram @ centre),
        "background matvec a @ X": lambda: a @ stack,
        "threshold mask |g - b| > 25 (np.where + astype)": lambda: np.where(
            np.abs(flat - background) > MOTION_THRESHOLD, 255, 0
        ).astype(np.uint8),
        "warm AND fg + countNonZero + 8x4 tiles": lambda: (
            cv2.countNonZero(cv2.bitwise_and(warm, fg)),
            cv2.resize((cv2.bitwise_and(warm, fg) > 0).astype(np.float32), (8, 4), interpolation=cv2.INTER_AREA),
        ),
        "whole model.apply (all of the above + ring buffer write)": lambda: model.apply(gray),
    }
    for name, step in stages.items():
        print(f"{name:62s} {_ms(step):7.3f} ms")
    print(f"products shape {products.shape}")


if __name__ == "__main__":
    main()
