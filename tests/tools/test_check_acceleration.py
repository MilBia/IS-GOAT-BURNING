"""Tests for the hardware acceleration probe tool (#78)."""

import builtins
import sys
from types import SimpleNamespace

import pytest

from tools import check_acceleration as ca


def _block_import(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Makes ``import <name>`` raise ImportError."""
    monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = builtins.__import__

    def fake_import(mod: str, *args, **kwargs):
        if mod == name:
            raise ImportError(name)
        return real_import(mod, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_probe_cv2_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_import(monkeypatch, "cv2")
    info = ca.probe_cv2()
    assert info["version"] is None
    assert "not installed" in info["error"]


def test_probe_cv2_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    dnn = SimpleNamespace(
        DNN_BACKEND_OPENCV=3,
        DNN_TARGET_CPU=0,
        DNN_TARGET_OPENCL=1,
        getAvailableTargets=lambda backend: [0, 1] if backend == 3 else [],
    )
    device = SimpleNamespace(name=lambda: "Fake iGPU")
    ocl = SimpleNamespace(haveOpenCL=lambda: True, useOpenCL=lambda: False, Device=SimpleNamespace(getDefault=lambda: device))
    fake = SimpleNamespace(__version__="5.0.0", dnn=dnn, ocl=ocl, cuda=SimpleNamespace(getCudaEnabledDeviceCount=lambda: 0))
    monkeypatch.setitem(sys.modules, "cv2", fake)

    info = ca.probe_cv2()

    assert info["version"] == "5.0.0"
    assert info["dnn"] == {"DNN_BACKEND_OPENCV": ["DNN_TARGET_CPU", "DNN_TARGET_OPENCL"]}
    assert info["opencl"] == {"have": True, "use": False, "device": "Fake iGPU"}
    assert info["cuda_devices"] == 0


def test_probe_cv2_without_cuda_module_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        __version__="5.0.0", dnn=SimpleNamespace(), ocl=SimpleNamespace(haveOpenCL=lambda: False, useOpenCL=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "cv2", fake)
    info = ca.probe_cv2()
    assert info["cuda_devices"] is None
    assert info["opencl"]["device"] is None


def test_format_report_layout() -> None:
    cv2_info = {
        "version": "5.0.0",
        "dnn": {"DNN_BACKEND_OPENCV": ["DNN_TARGET_CPU"]},
        "opencl": {"have": False, "use": False, "device": None},
        "cuda_devices": None,
    }
    report = ca.format_report(cv2_info, "engine=cv2dnn accelerator=cpu")
    assert "cv2 version: 5.0.0" in report
    assert "dnn DNN_BACKEND_OPENCV: DNN_TARGET_CPU" in report
    assert "haveOpenCL=False useOpenCL=False" in report
    assert "onnx" not in report.lower()
    assert "EDGE__ACCELERATOR=auto -> engine=cv2dnn accelerator=cpu" in report
    assert "benchmark" not in report.lower()


def test_format_report_benchmarks() -> None:
    cv2_info = {"version": None, "error": "cv2 not installed"}
    benchmarks = {"cpu": {"ok": True, "median_ms": 12.34}, "cuda": {"ok": False, "error": "EdgeModelError: no cuda"}}
    report = ca.format_report(cv2_info, "x", benchmarks)
    assert "cv2: cv2 not installed" in report
    assert "cpu: ok, median 12.3 ms" in report
    assert "cuda: error: EdgeModelError: no cuda" in report


def test_benchmark_accelerator_reports_missing_model_as_error(tmp_path) -> None:
    result = ca.benchmark_accelerator("cpu", str(tmp_path / "missing.onnx"), 224)
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_main_exits_zero_and_prints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ca, "probe_cv2", lambda: {"version": None, "error": "cv2 not installed"})
    monkeypatch.setattr(ca, "probe_auto_selection", lambda: "engine=cv2dnn accelerator=cpu")
    lines: list[str] = []
    assert ca.main([], out=lines.append) == 0
    assert "== OpenCV ==" in lines[0]
    assert "== hybrid_edge selection ==" in lines[0]


def test_probe_cv2_without_dnn_module_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(__version__="4.0"))
    info = ca.probe_cv2()
    assert info["version"] == "4.0"
    assert "probe failed" in info["error"]
    assert "probe failed" in ca.format_report(info, "x")


def test_main_never_raises_on_unexpected_probe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(ca, "probe_cv2", boom)
    lines: list[str] = []
    assert ca.main([], out=lines.append) == 0
    assert "boom" in lines[0]


def test_main_with_model_benchmarks_every_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ca, "benchmark_accelerator", lambda _a, _m, _s: {"ok": True, "median_ms": 1.0})
    lines: list[str] = []
    assert ca.main(["--model", "m.onnx", "--size", "224"], out=lines.append) == 0
    for acc in ("auto", "cpu", "cuda", "opencl", "openvino", "openvino_gpu"):
        assert f"{acc}: ok" in lines[0]


def test_invalid_size_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        ca.main(["--size", "0"])
    assert exc.value.code == 2
