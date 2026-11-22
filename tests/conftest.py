"""Suite-wide test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _restore_ort_telemetry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Records ORT_DISABLE_TELEMETRY so the value ``_import_onnxruntime()`` sets is undone after each test (#207)."""
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "1")
