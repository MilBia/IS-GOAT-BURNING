"""Candidate persistence (P15) and evidence accumulation (P8) for the fire-rule harness (#134).

``BlobTracker`` follows warm-AND-moving components from check to check (box IoU) and exposes how long a
candidate has been present, how well its base stays anchored, whether it grows and how elongated it is.
``LeakyIntegrator`` is a leaky (exponential) accumulator of evidence on a tile grid, optionally gated by
the boxes of persistent tracks. Both are plain NumPy and run at one check per second.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import field

import numpy as np

HISTORY = 8
IOU_MIN = 0.05
MAX_GAP = 1  # a track survives this many missed checks
ANCHOR_DRIFT = 0.05  # allowed drift of the base point, in frame widths


@dataclass(frozen=True)
class TrackGate:
    """Which tracks count as fire-like candidates.

    Attributes:
        need: Number of checks (out of ``window``) in which the candidate must be present.
        window: Number of latest checks considered.
        anchored: Require the base point to stay within ``ANCHOR_DRIFT`` of where the track started.
        min_growth: Required ratio of the latest area to the oldest present area in the window (0 disables).
        min_elongation: Required box height / width (0 disables).
    """

    need: int = 3
    window: int = 3
    anchored: bool = False
    min_growth: float = 0.0
    min_elongation: float = 0.0


@dataclass
class Track:
    """A followed candidate."""

    box: np.ndarray
    anchor: tuple[float, float]
    areas: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY))
    missed: int = 0

    @property
    def base(self) -> tuple[float, float]:
        """Base point (bottom centre of the box)."""
        return float(self.box[0] + self.box[2] / 2), float(self.box[1] + self.box[3])

    def persistent_area(self, gate: TrackGate) -> float:
        """Returns the ``need``-th largest area of the last ``window`` checks, or 0 if the gate is not passed."""
        recent = list(self.areas)[-gate.window :]
        if len(recent) < gate.window:
            return 0.0
        value = sorted(recent, reverse=True)[gate.need - 1]
        if value <= 0.0 or self.areas[-1] <= 0.0:
            return 0.0
        if gate.anchored:
            base_x, base_y = self.base
            if np.hypot(base_x - self.anchor[0], base_y - self.anchor[1]) > ANCHOR_DRIFT:
                return 0.0
        if gate.min_growth > 0.0:
            present = [a for a in recent if a > 0.0]
            if self.areas[-1] < gate.min_growth * present[0]:
                return 0.0
        if gate.min_elongation > 0.0 and self.box[3] < gate.min_elongation * max(float(self.box[2]), 1e-9):
            return 0.0
        return float(value)


def _iou_matrix(tracks: list[Track], boxes: np.ndarray) -> np.ndarray:
    """IoU between track boxes and detection boxes (both ``x, y, w, h``)."""
    a = np.array([t.box for t in tracks], dtype=np.float64)[:, None, :]
    b = boxes.astype(np.float64)[None, :, :]
    x0 = np.maximum(a[..., 0], b[..., 0])
    y0 = np.maximum(a[..., 1], b[..., 1])
    x1 = np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
    y1 = np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter
    return inter / np.maximum(union, 1e-12)


class BlobTracker:
    """Greedy IoU tracker over the per-check blob tables from ``features.blob_table``."""

    def __init__(self) -> None:
        self.tracks: list[Track] = []

    def reset(self) -> None:
        """Forgets all tracks (stream restart)."""
        self.tracks = []

    def update(self, blobs: np.ndarray) -> list[Track]:
        """Advances one check.

        Args:
            blobs: ``(slots, 5)`` table; rows with zero area are padding.

        Returns:
            Tracks alive after the update.
        """
        live = blobs[blobs[:, 4] > 0]
        matched_tracks: set[int] = set()
        matched_blobs: set[int] = set()
        if self.tracks and len(live):
            iou = _iou_matrix(self.tracks, live[:, :4])
            for flat in np.argsort(-iou, axis=None):
                ti, bi = divmod(int(flat), iou.shape[1])
                if iou[ti, bi] < IOU_MIN:
                    break
                if ti in matched_tracks or bi in matched_blobs:
                    continue
                matched_tracks.add(ti)
                matched_blobs.add(bi)
                track = self.tracks[ti]
                track.box = live[bi, :4].copy()
                track.areas.append(float(live[bi, 4]))
                track.missed = 0
        survivors: list[Track] = []
        for ti, track in enumerate(self.tracks):
            if ti in matched_tracks:
                survivors.append(track)
                continue
            track.missed += 1
            track.areas.append(0.0)
            if track.missed <= MAX_GAP:
                survivors.append(track)
        for bi, row in enumerate(live):
            if bi in matched_blobs:
                continue
            track = Track(row[:4].copy(), (float(row[0] + row[2] / 2), float(row[1] + row[3])))
            track.areas.append(float(row[4]))
            survivors.append(track)
        self.tracks = survivors
        return survivors

    def score(self, gate: TrackGate) -> float:
        """Largest persistent area (fraction of the frame) among tracks passing ``gate``."""
        return max((t.persistent_area(gate) for t in self.tracks), default=0.0)

    def persistent_boxes(self, gate: TrackGate) -> list[np.ndarray]:
        """Boxes of the tracks passing ``gate``."""
        return [t.box for t in self.tracks if t.persistent_area(gate) > 0.0]


class LeakyIntegrator:
    """Exponential accumulation of per-tile evidence: ``E <- decay * E + (1 - decay) * x``.

    A sustained input ``x`` converges to ``x``, so the state never exceeds the largest sustained input and a
    single burst is attenuated by ``1 - decay``.
    """

    def __init__(self, decay: float, shape: tuple[int, int]) -> None:
        self.decay = decay
        self.state = np.zeros(shape, dtype=np.float64)

    def update(self, evidence: np.ndarray) -> float:
        """Adds one check of evidence (``(rows, cols)`` fractions) and returns the largest tile value."""
        self.state = self.decay * self.state + (1.0 - self.decay) * evidence
        return float(self.state.max())


def box_gate(boxes: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Boolean ``(rows, cols)`` map of the tiles overlapped by any of ``boxes`` (``x, y, w, h`` fractions)."""
    rows, cols = shape
    gate = np.zeros(shape, dtype=bool)
    for x, y, w, h in boxes:
        c0, c1 = int(np.floor(x * cols)), int(np.ceil((x + w) * cols))
        r0, r1 = int(np.floor(y * rows)), int(np.ceil((y + h) * rows))
        gate[max(r0, 0) : max(r1, r0 + 1), max(c0, 0) : max(c1, c0 + 1)] = True
    return gate
