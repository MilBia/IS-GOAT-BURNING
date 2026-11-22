"""Reports which hardware acceleration options are usable for the ``hybrid_edge`` verifier (#78).

Run with ``python -m tools.check_acceleration [--model PATH --size N]``. The tool only reports: it exits with 0
unless its arguments are invalid, and a missing optional package is reported as "not installed", never raised.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from collections.abc import Sequence
import statistics
import sys
import time
from typing import Any

import numpy as np

RUNS = 20

_BACKEND_NAMES = ("DNN_BACKEND_OPENCV", "DNN_BACKEND_INFERENCE_ENGINE", "DNN_BACKEND_CUDA")
_TARGET_NAMES = (
    "DNN_TARGET_CPU",
    "DNN_TARGET_OPENCL",
    "DNN_TARGET_OPENCL_FP16",
    "DNN_TARGET_MYRIAD",
    "DNN_TARGET_VULKAN",
    "DNN_TARGET_FPGA",
    "DNN_TARGET_CUDA",
    "DNN_TARGET_CUDA_FP16",
    "DNN_TARGET_HDDL",
    "DNN_TARGET_NPU",
    "DNN_TARGET_CPU_FP16",
)


def probe_cv2() -> dict[str, Any]:
    """Collects OpenCV version, DNN backend/target pairs, OpenCL and CUDA information.

    Returns:
        A dictionary; every probe that fails yields ``None`` or an ``error`` entry instead of raising.
    """
    info: dict[str, Any] = {"version": None, "dnn": {}, "opencl": None, "cuda_devices": None}
    try:
        import cv2
    except ImportError:
        info["error"] = "cv2 not installed"
        return info
    info["version"] = getattr(cv2, "__version__", "unknown")
    try:
        _probe_cv2_details(cv2, info)
    except Exception as e:  # e.g. cv2.dnn missing: report, never raise
        info["error"] = f"probe failed: {type(e).__name__}: {e}"
    return info


def _probe_cv2_details(cv2: Any, info: dict[str, Any]) -> None:
    """Fills ``info`` with DNN, OpenCL and CUDA data (may raise; the caller handles it)."""
    target_names = {getattr(cv2.dnn, name): name for name in _TARGET_NAMES if hasattr(cv2.dnn, name)}
    for backend_name in _BACKEND_NAMES:
        backend = getattr(cv2.dnn, backend_name, None)
        if backend is None:
            continue
        try:
            targets = cv2.dnn.getAvailableTargets(backend)
        except Exception as e:
            info["dnn"][backend_name] = f"error: {e}"
            continue
        info["dnn"][backend_name] = [target_names.get(t, f"target#{t}") for t in targets]

    try:
        info["opencl"] = {
            "have": bool(cv2.ocl.haveOpenCL()),
            "use": bool(cv2.ocl.useOpenCL()),
            "device": _opencl_device_name(cv2),
        }
    except Exception as e:
        info["opencl"] = {"error": str(e)}
    try:
        info["cuda_devices"] = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        info["cuda_devices"] = None  # module absent (pip wheels) or built without CUDA


def _opencl_device_name(cv2: Any) -> str | None:
    """Returns the default OpenCL device name, or ``None`` when unavailable."""
    try:
        device = cv2.ocl.Device.getDefault()
        name = device.name()
        return str(name) if name else None
    except Exception:
        return None


def probe_auto_selection() -> str:
    """Describes what ``EDGE__ACCELERATOR=auto`` resolves to on this host.

    Returns:
        A one-line description of the selected engine and accelerator, or the reason it could not be resolved.
    """
    try:
        from is_goat_burning.config import Accelerator
        from is_goat_burning.fire_detection.edge_detector import cuda_available
        from is_goat_burning.fire_detection.edge_detector import opencl_available
        from is_goat_burning.fire_detection.edge_detector import select_backend

        plan = select_backend(Accelerator.AUTO, cuda_available=cuda_available(), opencl_available=opencl_available())
    except Exception as e:
        return f"unavailable ({type(e).__name__}: {e})"
    return f"engine={plan.engine} accelerator={plan.accelerator}"


_BENCH_ACCELERATORS = ("auto", "cpu", "cuda", "opencl", "openvino", "openvino_gpu")


def benchmark_accelerator(accelerator: str, model_path: str, size: int, runs: int = RUNS) -> dict[str, Any]:
    """Tries ``EdgeInferenceDetector`` with one accelerator and measures the median latency.

    Args:
        accelerator: One of ``_BENCH_ACCELERATORS``.
        model_path: Path to the ONNX classifier.
        size: Square model input side.
        runs: Number of timed inferences.

    Returns:
        ``{"ok": True, "median_ms": float}`` or ``{"ok": False, "error": str}``.
    """
    try:
        from is_goat_burning.config import EdgeSettings
        from is_goat_burning.fire_detection.edge_detector import EdgeInferenceDetector

        detector = EdgeInferenceDetector(EdgeSettings(model_path=model_path, accelerator=accelerator, input_size=size))
        frame = np.random.default_rng(0).integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            detector._infer(frame)
            timings.append((time.perf_counter() - start) * 1000.0)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "median_ms": statistics.median(timings)}


def format_report(cv2_info: dict[str, Any], auto: str, benchmarks: dict[str, dict[str, Any]] | None = None) -> str:
    """Renders the probe results as plain text.

    Args:
        cv2_info: Result of ``probe_cv2``.
        auto: Result of ``probe_auto_selection``.
        benchmarks: Optional per-accelerator results of ``benchmark_accelerator``.

    Returns:
        The report text (no trailing newline).
    """
    lines = ["== OpenCV =="]
    if cv2_info.get("version") is None:
        lines.append(f"cv2: {cv2_info.get('error', 'not installed')}")
    else:
        lines.append(f"cv2 version: {cv2_info['version']}")
        if cv2_info.get("error"):
            lines.append(f"cv2: {cv2_info['error']}")
        for backend, targets in cv2_info["dnn"].items():
            shown = ", ".join(targets) if isinstance(targets, list) else targets
            lines.append(f"dnn {backend}: {shown or 'no targets'}")
        ocl = cv2_info.get("opencl") or {}
        if "error" in ocl:
            lines.append(f"opencl: error: {ocl['error']}")
        else:
            lines.append(f"opencl: haveOpenCL={ocl.get('have')} useOpenCL={ocl.get('use')} device={ocl.get('device')}")
        devices = cv2_info.get("cuda_devices")
        lines.append(f"cuda devices (cv2.cuda): {devices if devices is not None else 'n/a (no cv2.cuda module)'}")
    lines.append("")
    lines.append("== hybrid_edge selection ==")
    lines.append(f"EDGE__ACCELERATOR=auto -> {auto}")
    if benchmarks:
        lines.append("")
        lines.append(f"== Model benchmark (median of {RUNS} runs, includes preprocessing) ==")
        lines.append("(1080p frame resize + normalisation are timed too: not comparable to model-only numbers.)")
        for accelerator, result in benchmarks.items():
            if result["ok"]:
                lines.append(f"{accelerator}: ok, median {result['median_ms']:.1f} ms")
            else:
                lines.append(f"{accelerator}: error: {result['error']}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace. ``SystemExit(2)`` is raised by argparse on invalid arguments.
    """
    parser = argparse.ArgumentParser(prog="python -m tools.check_acceleration", description=__doc__)
    parser.add_argument("--model", help="ONNX classifier to benchmark with every accelerator")
    parser.add_argument("--size", type=int, default=224, help="square model input side (default: 224)")
    args = parser.parse_args(argv)
    if args.size <= 0:
        parser.error("--size must be positive")
    return args


def main(argv: Sequence[str] | None = None, out: Callable[[str], object] = print) -> int:
    """Runs the probes and prints the report.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
        out: Output function.

    Returns:
        ``0`` always (invalid arguments exit with 2 through argparse).
    """
    args = parse_args(argv)
    try:
        benchmarks = None
        if args.model:
            benchmarks = {acc: benchmark_accelerator(acc, args.model, args.size) for acc in _BENCH_ACCELERATORS}
        out(format_report(probe_cv2(), probe_auto_selection(), benchmarks))
    except Exception as e:  # the tool only reports; it must not fail
        out(f"check_acceleration failed unexpectedly: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
