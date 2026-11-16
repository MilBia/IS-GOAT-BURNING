"""Low-rank + sparse foreground masks for the warm-AND-moving rule (#140: eigenbackground, GoDec, RPCA).

PRE-REGISTRATION: see the section at the end of this docstring (frozen in its own commit, BEFORE any feature of these models
was extracted and before any ``compare``).

Models (all on the 8-bit grey frame at the working resolution, unblurred; one state per clip; nothing else tuned). The window
is the ``N`` previous checks (at least ``MIN_SAMPLES`` = 3 of them, else the mask is empty). Every model ends in the same
decision as the median control of #139: ``fg = |x_t - background_t| > MOTION_THRESHOLD`` (25 grey levels), so only the
*estimate of the background* differs between the methods.

* ``eig`` (eigenbackground, Oliver et al. 2000): mean + the ``k`` leading principal components of the ``N`` previous frames;
  the current frame is projected onto that subspace (current frame NOT part of the fit). Fast path: everything is expressed
  through the ``N x N`` Gram matrix, the background is a linear combination of the window frames (``background = X^T a``).
* ``godec`` (Zhou & Tao 2011): the ``N + 1`` columns (window + current) are decomposed as ``D = L + S + G`` with
  ``rank(L) <= r`` and ``S`` hard-thresholded at ``MOTION_THRESHOLD`` (the l0-proximal operator, i.e. the exact sparse step of
  GoDec with the cardinality given by the threshold); ``L`` by a randomised range finder (``Y = (D - S) A``,
  ``L = Q Q^T (D - S)``, ``A`` Gaussian ``P x r``, fixed seed, ``GODEC_POWER`` = 2 power steps: one random projection is
  dominated by the outlier row), ``GODEC_ITERATIONS`` = 3 alternations.
* ``rpca`` (principal component pursuit by inexact ALM, Lin et al. 2010): the same ``N + 1`` columns,
  ``lambda = 1 / sqrt(max(N + 1, P))``, ``mu_0 = 1.25 / ||D||_2``, ``rho = 1.5``, stop at relative residual ``1e-4`` or
  ``RPCA_MAX_ITERATIONS`` = 50 (the pre-registered candidate is the *converged* one; a capped variant is a sensitivity run: an
  iteration cap changes the algorithm, because the sparse shrink threshold ``lambda / mu`` reaches pixel scale only after
  many iterations). SVD of the ``(N+1) x P`` matrix through the ``(N+1) x (N+1)`` Gram matrix.
* ``mean`` (control, ``eig`` with ``k = 0``): the plain mean of the window.

Per check and model ``v`` the extractor stores the columns of #139: ``bgwm_v`` = count(warm AND fg), ``bgand_v`` =
count(warm AND fg AND motion25), ``bgmv_v`` = count(fg), ``tile_bg_v`` = 8x4 tile fractions of warm AND fg; plus, for ``rpca``,
``lriter_v`` = iterations used (cost log).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from is_goat_burning.fire_detection.detectors import MAX_PIXEL_VALUE

MOTION_THRESHOLD = 25
MIN_SAMPLES = 3
GODEC_ITERATIONS = 3
GODEC_POWER = 2
RPCA_MAX_ITERATIONS = 50
RPCA_TOLERANCE = 1e-4
RPCA_RHO = 1.5
SEED = 0
EPS = 1e-9


@dataclass(frozen=True)
class LowRankSpec:
    """One low-rank model.

    Attributes:
        kind: ``eig``, ``godec``, ``rpca`` or ``mean``.
        window: Number of previous checks in the window (``N``).
        rank: Rank ``k`` (``eig``) or ``r`` (``godec``); unused for ``rpca`` / ``mean``.
        iterations: Alternations (``godec``) or maximum iterations (``rpca``); 0 = the module default.
    """

    kind: str
    window: int
    rank: int = 0
    iterations: int = 0


CANDIDATE_VARIANTS: dict[str, LowRankSpec] = {
    "lr_eig1_n5": LowRankSpec("eig", 5, 1),
    "lr_eig1_n9": LowRankSpec("eig", 9, 1),
    "lr_eig2_n9": LowRankSpec("eig", 9, 2),
    "lr_godec1_n5": LowRankSpec("godec", 5, 1),
    "lr_godec1_n9": LowRankSpec("godec", 9, 1),
    "lr_godec2_n9": LowRankSpec("godec", 9, 2),
    "lr_rpca_n5": LowRankSpec("rpca", 5),
    "lr_rpca_n9": LowRankSpec("rpca", 9),
}
CONTROL_VARIANTS: dict[str, LowRankSpec] = {"lr_mean_n9": LowRankSpec("mean", 9)}
SENSITIVITY_VARIANTS: dict[str, LowRankSpec] = {
    "lr_eig1_n15": LowRankSpec("eig", 15, 1),
    "lr_godec1_n15": LowRankSpec("godec", 15, 1),
    "lr_godec1_n9_t6": LowRankSpec("godec", 9, 1, 6),
    "lr_rpca_n9_c5": LowRankSpec("rpca", 9, 0, 5),
}
ALL_VARIANTS = {**CANDIDATE_VARIANTS, **CONTROL_VARIANTS, **SENSITIVITY_VARIANTS}


def _fg(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """``uint8`` 0 / 255 mask of ``|frame - background| > MOTION_THRESHOLD`` (flat arrays)."""
    return np.where(np.abs(frame - background) > MOTION_THRESHOLD, MAX_PIXEL_VALUE, 0).astype(np.uint8)


class _Window:
    """Ring buffer of the previous frames (flat ``float32``, shifted by the first frame) with their Gram matrix."""

    def __init__(self, size: int, shift: bool) -> None:
        self.size = size
        self.shift = shift
        self.count = 0
        self.reference: np.ndarray | None = None
        self.frames: np.ndarray | None = None
        self.gram = np.zeros((size, size), dtype=np.float64)

    def prepare(self, gray: np.ndarray) -> np.ndarray:
        """Flat float32 copy of ``gray`` (shifted by the reference frame for the Gram based models)."""
        flat = gray.reshape(-1).astype(np.float32)
        if not self.shift:
            return flat
        if self.reference is None:
            self.reference = flat.copy()
        return flat - self.reference

    def view(self) -> np.ndarray:
        """The ``m x P`` stack of the frames held (order irrelevant)."""
        assert self.frames is not None
        return self.frames[: min(self.count, self.size)]

    def push(self, flat: np.ndarray, products: np.ndarray | None = None) -> None:
        """Adds a frame; ``products`` = ``view() @ flat`` computed before the call (saves one pass)."""
        if self.frames is None:
            self.frames = np.zeros((self.size, flat.size), dtype=np.float32)
        slot = self.count % self.size
        m = min(self.count, self.size)
        if self.shift:
            row = self.view() @ flat if products is None else products
            self.gram[slot, :m] = row
            self.gram[:m, slot] = row
            self.gram[slot, slot] = float(flat @ flat)
        self.frames[slot] = flat
        self.count += 1


def eig_background(window: np.ndarray, gram: np.ndarray, products: np.ndarray, rank: int) -> np.ndarray:
    """Eigenbackground of the current frame from the window Gram matrix (fast path; ``background = window^T a``).

    Args:
        window: ``m x P`` previous frames (shifted by a constant, which leaves the result unchanged).
        gram: ``m x m`` Gram matrix ``window @ window.T``.
        products: ``window @ current`` (``m`` vector).
        rank: Number of principal components ``k`` (0 = the mean).

    Returns:
        The background estimate (flat ``float32``, same shift as ``window``).
    """
    m = window.shape[0]
    a = np.full(m, 1.0 / m)
    rank = min(rank, m - 1)
    if rank > 0:
        centre = np.eye(m) - 1.0 / m
        values, vectors = np.linalg.eigh(centre @ gram @ centre)
        order = np.argsort(values)[::-1][:rank]
        values, vectors = values[order], vectors[:, order]
        keep = values > 1e-6 * max(values[0], 1.0)
        if keep.any():
            weights = (centre @ vectors[:, keep]) / np.sqrt(values[keep])
            coefficients = weights.T @ (products - gram.sum(axis=1) / m)
            a = a + weights @ coefficients
    return (a.astype(np.float32) @ window).astype(np.float32)


def eig_background_naive(window: np.ndarray, current: np.ndarray, rank: int) -> np.ndarray:
    """Reference implementation by a full SVD of the centred window (used by the tests)."""
    x = window.astype(np.float64)
    mean = x.mean(axis=0)
    if rank == 0:
        return mean
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    basis = vt[: min(rank, len(s))][s[: min(rank, len(s))] > 1e-6]
    centred = current.astype(np.float64) - mean
    return mean + basis.T @ (basis @ centred)


def godec_low_rank(
    data: np.ndarray, rank: int, iterations: int, projection: np.ndarray, threshold: float = MOTION_THRESHOLD
) -> np.ndarray:
    """GoDec: low-rank part ``L`` of ``data`` (``M x P``) with a hard-thresholded sparse part.

    Args:
        data: ``M x P`` float32 matrix (window frames + current frame as the last row).
        rank: Rank of ``L``.
        iterations: Number of ``L`` / ``S`` alternations.
        projection: ``P x rank`` Gaussian test matrix.
        threshold: Hard threshold of the sparse step.

    Returns:
        ``L`` (``M x P``).
    """
    sparse = np.zeros_like(data)
    low = data
    for _ in range(iterations):
        residual = data - sparse
        y = residual @ projection
        for _ in range(GODEC_POWER):  # power scheme of the original GoDec: a single random projection is dominated by outliers
            y = y / np.maximum(np.linalg.norm(y, axis=0), EPS)
            y = residual @ (residual.T @ y)
        q, _ = np.linalg.qr(y)
        low = q.astype(np.float32) @ (q.T.astype(np.float32) @ residual)
        diff = data - low
        sparse = np.where(np.abs(diff) > threshold, diff, np.float32(0))
    return low


def rpca_low_rank(data: np.ndarray, max_iterations: int = RPCA_MAX_ITERATIONS) -> tuple[np.ndarray, int]:
    """Principal component pursuit by inexact ALM; SVD through the small Gram matrix.

    Args:
        data: ``M x P`` matrix (float64), window + current frame as the last row.
        max_iterations: Iteration cap.

    Returns:
        ``(L, iterations_used)``.
    """
    m, p = data.shape
    lam = 1.0 / np.sqrt(max(m, p))
    norm2 = float(np.sqrt(max(np.linalg.eigvalsh(data @ data.T)[-1], EPS)))
    dual = max(norm2, float(np.abs(data).max()) / lam)
    y = data / dual
    mu = 1.25 / norm2
    norm_data = float(np.linalg.norm(data))
    sparse = np.zeros_like(data)
    low = np.zeros_like(data)
    used = 0
    for used in range(1, max_iterations + 1):
        target = data - sparse + y / mu
        values, vectors = np.linalg.eigh(target @ target.T)
        sigma = np.sqrt(np.maximum(values, 0.0))
        shrink = np.where(sigma > 1.0 / mu, (sigma - 1.0 / mu) / np.maximum(sigma, EPS), 0.0)
        low = ((vectors * shrink) @ vectors.T) @ target
        residual = data - low + y / mu
        sparse = np.sign(residual) * np.maximum(np.abs(residual) - lam / mu, 0.0)
        z = data - low - sparse
        y = y + mu * z
        mu = mu * RPCA_RHO
        if np.linalg.norm(z) / norm_data < RPCA_TOLERANCE:
            break
    return low, used


class _LowRankModel:
    """Foreground mask of one model; call :meth:`apply` once per check, in order."""

    def __init__(self, spec: LowRankSpec) -> None:
        self.spec = spec
        self.window = _Window(spec.window, shift=spec.kind in ("eig", "mean"))
        self.iterations_used = 0
        self._projection: np.ndarray | None = None

    def apply(self, gray: np.ndarray) -> np.ndarray:
        """Returns the ``uint8`` foreground mask (0 / 255, shape of ``gray``) and updates the window."""
        spec = self.spec
        flat = self.window.prepare(gray)
        if self.window.count < MIN_SAMPLES:
            fg = np.zeros(flat.size, dtype=np.uint8)
            self.iterations_used = 0
            self.window.push(flat)
            return fg.reshape(gray.shape)
        stack = self.window.view()
        if spec.kind in ("eig", "mean"):
            products = stack @ flat
            background = eig_background(stack, self.window.gram[: len(stack), : len(stack)], products, spec.rank)
            fg = _fg(flat, background)
            self.window.push(flat, products)
        elif spec.kind == "godec":
            if self._projection is None:
                self._projection = np.random.default_rng(SEED).standard_normal((flat.size, spec.rank)).astype(np.float32)
            data = np.vstack([stack, flat[None, :]])
            low = godec_low_rank(data, spec.rank, spec.iterations or GODEC_ITERATIONS, self._projection)
            fg = _fg(flat, low[-1])
            self.window.push(flat)
        elif spec.kind == "rpca":
            data = np.vstack([stack, flat[None, :]]).astype(np.float64)
            low, self.iterations_used = rpca_low_rank(data, spec.iterations or RPCA_MAX_ITERATIONS)
            fg = _fg(flat, low[-1].astype(np.float32))
            self.window.push(flat)
        else:
            raise ValueError(spec.kind)
        return fg.reshape(gray.shape)


class LowRankModels:
    """Per-stream state of a set of low-rank models (same interface as ``bgmodel.BgModels``)."""

    def __init__(self, variants: tuple[str, ...] = tuple(CANDIDATE_VARIANTS)) -> None:
        self.variants = variants
        self._models = {name: _LowRankModel(ALL_VARIANTS[name]) for name in variants}

    def masks(self, gray: np.ndarray) -> dict[str, np.ndarray]:
        """Foreground masks of all models for this check (updates every model)."""
        return {name: model.apply(gray) for name, model in self._models.items()}

    def update(self, gray: np.ndarray, color: np.ndarray, motion25: np.ndarray | None) -> dict[str, float | np.ndarray]:
        """Processes one check; returns the ``bgwm_`` / ``bgand_`` / ``bgmv_`` / ``tile_bg_`` / ``lriter_`` columns."""
        from tools.fire_rule_eval.bgmodel import BG_TILE_GRID

        out: dict[str, float | np.ndarray] = {}
        for name, fg in self.masks(gray).items():
            joint = cv2.bitwise_and(color, fg)
            out[f"bgwm_{name}"] = int(cv2.countNonZero(joint))
            out[f"bgand_{name}"] = (
                out[f"bgwm_{name}"] if motion25 is None else int(cv2.countNonZero(cv2.bitwise_and(joint, motion25)))
            )
            out[f"bgmv_{name}"] = int(cv2.countNonZero(fg))
            out[f"tile_bg_{name}"] = cv2.resize(
                (joint > 0).astype(np.float32), BG_TILE_GRID, interpolation=cv2.INTER_AREA
            ).reshape(-1)
            out[f"lriter_{name}"] = self._models[name].iterations_used
        return out
