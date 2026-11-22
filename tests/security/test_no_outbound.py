"""Local inference must not phone home (#210, background in #207).

Each test runs ``no_outbound_child.py`` in a fresh Python process with an empty temporary ``HOME``,
``XDG_CACHE_HOME``, ``TMPDIR`` and working directory, *without* any telemetry opt-out variable and without CI
markers (``CI_DETECTION_VARIABLES``: telemetry clients go quiet on CI), so only the app's
own guard (``_block_openvino_telemetry``) can keep a library quiet. The child builds
``EdgeInferenceDetector`` on a tiny generated ONNX model for every backend installed here and runs a few inferences.

Evidence of a leak, checked by the parent:
  * a non-local DNS lookup / connect / sendto seen by the child's Python-level socket guard;
  * any file created in the temporary directories (e.g. OpenVINO's telemetry ``~/intel/openvino_ga_cid``) — this is
    what catches native code;
  * a child process still alive after the wait (OpenVINO's telemetry uploads from a ``multiprocessing`` forkserver,
    which the in-process guard does not see).
If ``unshare -rn`` works, the child also runs in an empty network namespace, so native code cannot reach the network
even when it is not observed. Limitation: an uploader delayed beyond ``NO_OUTBOUND_WAIT_SECONDS`` (default 3 s) that
writes nothing to disk would go unnoticed outside the namespace.

The ``raw_*`` controls use the libraries *without* the app's guards and must be caught, proving the detection works;
the library ones run only inside the network namespace (otherwise they would really send telemetry).
``raw_openvino`` needs ``openvino-telemetry``, which the documented install leaves out, so it is skipped without it
unless ``NO_OUTBOUND_REQUIRE_OV_TELEMETRY`` is set (non-empty): then a missing package fails the test (CI's step that
installs the package to check the in-code block sets it).
"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import pytest

from tests.fire_detection.test_edge_detector import _build_classifier_onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD_SCRIPT = Path(__file__).with_name("no_outbound_child.py")
WAIT_SECONDS = float(os.environ.get("NO_OUTBOUND_WAIT_SECONDS", "3"))
# CI sets this so a runner without unprivileged user namespaces fails instead of silently losing the native-code block.
REQUIRE_NETNS = os.environ.get("NO_OUTBOUND_REQUIRE_NETNS") == "1"
# CI sets this in the step that installs openvino-telemetry, so the OpenVINO leak control cannot silently skip.
REQUIRE_OV_TELEMETRY = bool(os.environ.get("NO_OUTBOUND_REQUIRE_OV_TELEMETRY"))
CHILD_TIMEOUT_SECONDS = 120
DETECTOR_RESULTS = [True, False, True, False]  # red, blue, red, blue frames
FILES_ALLOWED_IN_WORKDIR = {"model.onnx", ".env.tests"}
# Telemetry clients stay silent when they detect a CI system, which would make the leak controls (and the guards)
# untestable in CI: openvino-telemetry checks CI, TF_BUILD and JENKINS_URL, and other clients check similar names. They
# are removed from the child so libraries behave as on a user's machine.
CI_DETECTION_VARIABLES = {
    "CI",
    "TF_BUILD",
    "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "JENKINS_URL",
    "CODEBUILD_BUILD_ID",
    "BUILDKITE",
    "TEAMCITY_VERSION",
    "APPVEYOR",
    "BITBUCKET_BUILD_NUMBER",
}


def _installed(module: str) -> bool:
    """Checks whether a module is installed without importing it (importing would run its telemetry here)."""
    return importlib.util.find_spec(module) is not None


def _netns_available() -> bool:
    """Checks whether an unprivileged empty network namespace can be created (``unshare -rn``)."""
    if shutil.which("unshare") is None:
        return False
    return subprocess.run(["unshare", "-rn", "true"], capture_output=True, check=False).returncode == 0


NETNS = _netns_available()

requires_openvino = pytest.mark.skipif(not _installed("openvino"), reason="openvino is not installed")
# The raw_* controls really emit telemetry: run them only where the empty network namespace keeps it from leaving.
requires_netns = pytest.mark.skipif(not NETNS, reason="unshare -rn unavailable: the leak control would really send telemetry")


def _child_env(home: Path, cache: Path, tmp: Path) -> dict[str, str]:
    """Builds the child environment: temporary home/cache/tmp, repo on the path, no telemetry opt-outs, no CI markers."""
    dropped = {"DO_NOT_TRACK", "PYTHONPATH", *CI_DETECTION_VARIABLES}
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("XDG_") and "TELEMETRY" not in key and key not in dropped
    }
    env.update(HOME=str(home), XDG_CACHE_HOME=str(cache), TMPDIR=str(tmp), PYTHONPATH=str(REPO_ROOT))
    return env


def _created_files(root: Path) -> list[str]:
    """Lists everything below ``root`` (relative paths) except the files the test put in the work dir itself."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if str(p.relative_to(root)) not in FILES_ALLOWED_IN_WORKDIR)


def _run_child(mode: str, tmp_path: Path) -> tuple[dict, list[str]]:
    """Runs the child for ``mode`` and returns its JSON report and the files it left in the temporary directories."""
    if not NETNS:
        message = "unshare -rn is unavailable: native code is checked only via files and child processes"
        if REQUIRE_NETNS:
            pytest.fail(message)
        warnings.warn(message, stacklevel=2)
    dirs = {name: tmp_path / name for name in ("home", "cache", "tmp", "work")}
    for directory in dirs.values():
        directory.mkdir()
    (dirs["work"] / "model.onnx").write_bytes(_build_classifier_onnx(8))
    shutil.copy(REPO_ROOT / ".env.tests", dirs["work"] / ".env.tests")  # app settings without a developer's .env
    command = [sys.executable, str(CHILD_SCRIPT), mode, "model.onnx", str(WAIT_SECONDS)]
    proc = subprocess.run(
        (["unshare", "-rn", *command] if NETNS else command),
        cwd=dirs["work"],
        env=_child_env(dirs["home"], dirs["cache"], dirs["tmp"]),
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        check=False,
    )
    assert proc.returncode == 0, f"child failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    files = [f"{name}/{path}" for name, directory in dirs.items() for path in _created_files(directory)]
    return report, files


@pytest.mark.parametrize(
    "mode",
    [
        "cv2dnn",
        pytest.param("openvino", marks=requires_openvino),
    ],
)
def test_edge_inference_makes_no_outbound_connections(mode, tmp_path):
    """The app's edge backends run fully offline: no connections, no files, no leftover processes."""
    report, files = _run_child(mode, tmp_path)

    assert report["results"] == DETECTOR_RESULTS, "inference did not run as expected"
    assert report["attempts"] == [], "network connection attempts"
    assert files == [], "files written outside the model directory (telemetry ids / caches)"
    assert report["children"] == [], "child processes left running after inference"


def test_detects_python_connection_attempt(tmp_path):
    """Control: the socket guard records (and refuses) a plain Python connection attempt."""
    report, _ = _run_child("raw_socket", tmp_path)

    assert any("telemetry.invalid" in attempt for attempt in report["attempts"]), report


@pytest.mark.skipif(
    not _installed("openvino_telemetry") and not REQUIRE_OV_TELEMETRY,
    reason="openvino-telemetry is not installed (as documented)",
)
@requires_openvino
@requires_netns
def test_detects_raw_openvino_telemetry(tmp_path):
    """Control: importing openvino without the ``sys.modules`` block, with openvino-telemetry installed, leaks."""
    if not _installed("openvino_telemetry"):
        pytest.fail("NO_OUTBOUND_REQUIRE_OV_TELEMETRY is set but openvino-telemetry is not installed")
    report, files = _run_child("raw_openvino", tmp_path)

    assert report["results"] == [True]
    assert any(f.startswith("home/intel") for f in files), f"leak not detected; files: {files}"
    assert report["children"], "telemetry forkserver not detected"
