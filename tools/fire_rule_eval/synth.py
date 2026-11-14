"""Synthetic positives: real flames composited onto a real (fire-free) background video.

The result keeps the background's sensor noise and compression while giving full control over how
much of the frame the fire occupies. Positives built this way are **synthetic** and must be labelled
as such in any report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FLAME_HUE_MAX = 35
FLAME_SATURATION_MIN = 60
FLAME_VALUE_MIN = 170
ALPHA_CLOSE_KERNEL = 9
ALPHA_BLUR = (7, 7)
BBOX_ALPHA_MIN = 0.1
BBOX_CLOSE_KERNEL = 21
SAMPLE_FRAMES = 40


@dataclass(frozen=True)
class SynthOptions:
    """Compositing parameters.

    Attributes:
        fraction: Target flame area as a fraction of the background frame area.
        anchor_x: Horizontal anchor of the flame base, as a fraction of frame width.
        anchor_y: Vertical anchor of the flame base, as a fraction of frame height.
        seconds: Output duration.
        fire_start_s: Offset into the fire clip to start from (skips ignition / camera moves).
        onset_s: Time at which the fire appears in the output (earlier frames are untouched background).
    """

    fraction: float = 0.01
    anchor_x: float = 0.52
    anchor_y: float = 0.76
    seconds: float = 20.0
    fire_start_s: float = 0.0
    onset_s: float = 0.0


def flame_alpha(frame: np.ndarray) -> np.ndarray:
    """Returns a soft alpha mask (0..1) of bright flame pixels, white-hot cores included."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = (hue <= FLAME_HUE_MAX) & (saturation > FLAME_SATURATION_MIN)
    white_hot = saturation <= FLAME_SATURATION_MIN
    mask = ((warm | white_hot) & (value >= FLAME_VALUE_MIN)).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ALPHA_CLOSE_KERNEL, ALPHA_CLOSE_KERNEL))
    )
    return cv2.GaussianBlur(mask.astype(np.float32), ALPHA_BLUR, 0)


def _read_frames(path: Path, start_s: float, count: int, out_fps: float) -> list[np.ndarray]:
    """Reads ``count`` frames resampled to ``out_fps``, looping the clip when it is too short."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or out_fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = min(round(start_s * fps), max(total - 1, 0))
    frames: list[np.ndarray] = []
    index = first
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    for k in range(count):
        wanted = first + round(k * fps / out_fps)
        wanted = first + (wanted - first) % max(total - first, 1)
        if wanted < index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, wanted)
            index = wanted
        ok = True
        frame = None
        while index <= wanted and ok:
            ok, frame = cap.read()
            index += 1
        if frame is None:
            break
        frames.append(frame)
    cap.release()
    return frames


def _flame_box(frames: list[np.ndarray]) -> tuple[slice, slice, float]:
    """Finds the bounding box of persistent flames and the median flame area inside it."""
    sample = frames[:: max(len(frames) // SAMPLE_FRAMES, 1)]
    alphas = [flame_alpha(f) for f in sample]
    persistent = (np.mean(alphas, axis=0) > BBOX_ALPHA_MIN).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BBOX_CLOSE_KERNEL, BBOX_CLOSE_KERNEL))
    persistent = cv2.morphologyEx(persistent, cv2.MORPH_CLOSE, close_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(persistent, connectivity=8)
    if count <= 1:
        raise ValueError("no persistent flame-coloured region found in the fire clip")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == largest)
    box_y, box_x = slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)
    area = float(np.median([a[box_y, box_x].sum() for a in alphas]))
    return box_y, box_x, area


def composite_video(background: Path, fire: Path, out_path: Path, options: SynthOptions) -> int:
    """Writes a composite video and returns the number of frames written.

    Args:
        background: Fire-free background video (its resolution and frame rate are kept).
        fire: Video containing real flames.
        out_path: Output path (``.mp4``).
        options: Compositing parameters.

    Returns:
        The number of frames written.
    """
    cap = cv2.VideoCapture(str(background))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(options.seconds * fps)
    fire_frames = _read_frames(fire, options.fire_start_s, count, fps)
    box_y, box_x, flame_area = _flame_box(fire_frames)
    scale = float(np.sqrt(options.fraction * width * height / max(flame_area, 1.0)))
    patch_w, patch_h = max(int((box_x.stop - box_x.start) * scale), 2), max(int((box_y.stop - box_y.start) * scale), 2)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    x0 = int(options.anchor_x * width) - patch_w // 2
    y0 = int(options.anchor_y * height) - patch_h
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + patch_w, width), min(y0 + patch_h, height)
    written = 0
    for index, fire_frame in enumerate(fire_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if index / fps >= options.onset_s and xe > xs and ye > ys:
            patch = cv2.resize(fire_frame[box_y, box_x], (patch_w, patch_h), interpolation=cv2.INTER_AREA)
            alpha_full = flame_alpha(fire_frame)[box_y, box_x]
            alpha = cv2.resize(alpha_full, (patch_w, patch_h), interpolation=cv2.INTER_AREA)[..., None]
            sub_patch = patch[ys - y0 : ye - y0, xs - x0 : xe - x0]
            sub_alpha = alpha[ys - y0 : ye - y0, xs - x0 : xe - x0]
            frame[ys:ye, xs:xe] = (sub_alpha * sub_patch + (1 - sub_alpha) * frame[ys:ye, xs:xe]).astype(np.uint8)
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    return written
