"""Latency benchmark of ONNX classifiers across inference backends.

Backends (each optional, skipped with a note when unavailable): cv2.dnn CPU, cv2.dnn OpenCL / OpenCL FP16,
onnxruntime CPU and OpenVINO CPU / GPU. Reports median and p95 latency plus the max abs output difference
versus onnxruntime CPU on the same random input, as a Markdown table.

Usage:
    python -m tools.edge_model.bench model.onnx [model2.onnx ...] --runs 40 --warmup 5 --threads 1 4
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

Runner = Callable[[], np.ndarray]


@dataclass
class Result:
    """One benchmark row.

    Attributes:
        backend: Backend label, e.g. ``ort-cpu/4thr``.
        median_ms: Median latency in milliseconds, or None when skipped.
        p95_ms: 95th percentile latency in milliseconds, or None when skipped.
        max_diff: Max abs output difference versus ORT CPU, or None.
        note: Skip or error note.
    """

    backend: str
    median_ms: float | None = None
    p95_ms: float | None = None
    max_diff: float | None = None
    note: str = ""


def measure(fn: Runner, runs: int, warmup: int) -> tuple[float, float]:
    """Time a callable.

    Args:
        fn: Zero-argument callable performing one inference.
        runs: Number of timed iterations.
        warmup: Number of untimed warm-up iterations.

    Returns:
        Tuple of (median, p95) latency in milliseconds.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    arr = np.array(times)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def _run(label: str, factory: Callable[[], Runner], ref: np.ndarray | None, args: argparse.Namespace) -> Result:
    """Build a runner, measure it and compare to the reference output; never raises."""
    try:
        fn = factory()
        out = np.asarray(fn()).reshape(-1)
        diff = None if ref is None else float(np.abs(out - ref.reshape(-1)).max())
        med, p95 = measure(fn, args.runs, args.warmup)
        return Result(label, med, p95, diff)
    except Exception as exc:  # noqa: BLE001 - backend failures must not abort the benchmark
        return Result(label, note=f"ERR {type(exc).__name__}: {str(exc).splitlines()[0][:120] if str(exc) else ''}")


def bench_model(path: Path, args: argparse.Namespace) -> tuple[tuple[int, ...], list[Result]]:
    """Benchmark one model on every available backend.

    Args:
        path: Path to the ONNX model.
        args: Parsed CLI arguments (runs, warmup, threads).

    Returns:
        Tuple of (input shape, list of results).
    """
    import onnxruntime as ort

    probe = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = probe.get_inputs()[0]
    shape = tuple(d if isinstance(d, int) else 1 for d in inp.shape)
    x = np.random.default_rng(0).random(shape, dtype=np.float32)
    results: list[Result] = []

    def ort_factory(threads: int) -> Callable[[], Runner]:
        def make() -> Runner:
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            return lambda: sess.run(None, {inp.name: x})[0]

        return make

    ref = np.asarray(ort_factory(1)()()).copy()
    ref_note = f"ref output max abs = {np.abs(ref).max():.2e}"
    for th in args.threads:
        results.append(_run(f"ort-cpu/{th}thr", ort_factory(th), ref, args))

    results.extend(_bench_cv2(path, x, ref, args))
    results.extend(_bench_openvino(path, x, ref, args))
    results.insert(0, Result("(reference)", note=ref_note))
    return shape, results


def _bench_cv2(path: Path, x: np.ndarray, ref: np.ndarray, args: argparse.Namespace) -> list[Result]:
    """Benchmark cv2.dnn on CPU and (if present) OpenCL targets."""
    try:
        import cv2
    except ImportError:
        return [Result("cv2.dnn", note="skipped: opencv not importable")]

    def factory(target: int, threads: int) -> Callable[[], Runner]:
        def make() -> Runner:
            cv2.setNumThreads(threads)
            # OpenCV 5 defaults to the new graph engine which ignores DNN targets (OpenCL); force the classic one.
            if args.cv2_engine == "classic" and hasattr(cv2.dnn, "ENGINE_CLASSIC"):
                net = cv2.dnn.readNetFromONNX(str(path), cv2.dnn.ENGINE_CLASSIC)
            else:
                net = cv2.dnn.readNetFromONNX(str(path))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(target)

            def fn() -> np.ndarray:
                net.setInput(x)
                return net.forward()

            return fn

        return make

    out: list[Result] = []
    for th in args.threads:
        out.append(_run(f"cv2.dnn-cpu/{th}thr", factory(cv2.dnn.DNN_TARGET_CPU, th), ref, args))
    have_ocl = bool(cv2.ocl.haveOpenCL())
    if have_ocl:
        cv2.ocl.setUseOpenCL(True)
        have_ocl = bool(cv2.ocl.useOpenCL())
    if not have_ocl:
        out.append(Result("cv2.dnn-opencl", note="skipped: no OpenCL device (cv2.ocl.haveOpenCL false)"))
    else:
        dev = cv2.ocl.Device.getDefault().name()
        top = max(args.threads)
        out.append(_run(f"cv2.dnn-opencl ({dev})", factory(cv2.dnn.DNN_TARGET_OPENCL, top), ref, args))
        out.append(_run(f"cv2.dnn-opencl-fp16 ({dev})", factory(cv2.dnn.DNN_TARGET_OPENCL_FP16, top), ref, args))
    return out


def _bench_openvino(path: Path, x: np.ndarray, ref: np.ndarray, args: argparse.Namespace) -> list[Result]:
    """Benchmark OpenVINO runtime on CPU and GPU when the package is importable."""
    try:
        import openvino as ov
    except ImportError:
        return [Result("openvino", note="skipped: openvino not importable")]
    core = ov.Core()
    devices = core.available_devices
    out: list[Result] = []

    def factory(device: str, config: dict) -> Callable[[], Runner]:
        def make() -> Runner:
            compiled = core.compile_model(str(path), device, config)
            req = compiled.create_infer_request()
            return lambda: req.infer({0: x})[compiled.output(0)]

        return make

    for th in args.threads:
        if "CPU" in devices:
            out.append(_run(f"openvino-cpu/{th}thr", factory("CPU", {"INFERENCE_NUM_THREADS": th}), ref, args))
    if "CPU" not in devices:
        out.append(Result("openvino-cpu", note=f"skipped: no CPU device in {devices}"))
    if "GPU" in devices:
        out.append(_run("openvino-gpu", factory("GPU", {}), ref, args))
    else:
        out.append(Result("openvino-gpu", note=f"skipped: no GPU device (available: {devices})"))
    return out


def to_markdown(name: str, shape: tuple[int, ...], results: list[Result]) -> str:
    """Render results of one model as a Markdown table.

    Args:
        name: Model file name.
        shape: Input shape used.
        results: Benchmark rows.

    Returns:
        Markdown string.
    """
    lines = [
        f"### {name} (input {list(shape)})",
        "",
        "| backend | median ms | p95 ms | max abs diff vs ORT CPU | note |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results:
        med = "-" if r.median_ms is None else f"{r.median_ms:.2f}"
        p95 = "-" if r.p95_ms is None else f"{r.p95_ms:.2f}"
        diff = "-" if r.max_diff is None else f"{r.max_diff:.2e}"
        lines.append(f"| {r.backend} | {med} | {p95} | {diff} | {r.note} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", type=Path, help="ONNX model files")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--cv2-engine",
        choices=("classic", "new"),
        default="classic",
        help="OpenCV 5 DNN engine; the new engine ignores OpenCL targets (default: classic)",
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    for path in args.models:
        shape, results = bench_model(path, args)
        print(to_markdown(path.name, shape, results))


if __name__ == "__main__":
    main()
