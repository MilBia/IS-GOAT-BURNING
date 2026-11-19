"""Tests for scripts/entrypoint.sh recordings-directory resolution."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

SH = shutil.which("sh")
pytestmark = pytest.mark.skipif(SH is None, reason="sh is not available")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "entrypoint.sh"
STUBS = ("mkdir", "chown", "gosu")


@pytest.fixture
def run_entrypoint(tmp_path):
    """Return a runner that executes the entrypoint with stubbed system commands."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    for name in STUBS:
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\n')
        stub.chmod(0o755)

    def _run(**env_vars):
        env = {k: v for k, v in os.environ.items() if not k.startswith("VIDEO")}
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        env["RENDER_DEVICE"] = str(tmp_path / "no-such-device")
        env.update(env_vars)
        result = subprocess.run([SH, str(SCRIPT), "app-cmd"], env=env, capture_output=True, text=True, check=False)
        calls = log.read_text().splitlines() if log.exists() else []
        return result, calls

    return _run


def test_new_variable_is_used(run_entrypoint):
    """VIDEO__VIDEO_OUTPUT_DIRECTORY drives mkdir and chown."""
    result, calls = run_entrypoint(VIDEO__VIDEO_OUTPUT_DIRECTORY="/app/custom_vids")
    assert result.returncode == 0
    assert "chown nobody:nogroup /app/custom_vids" in calls
    assert "mkdir -p /app/custom_vids" in calls
    assert "deprecat" not in result.stderr.lower()


def test_legacy_variable_only_warns(run_entrypoint):
    """The legacy name still works but prints a deprecation warning."""
    result, calls = run_entrypoint(VIDEO_OUTPUT_DIRECTORY="/app/legacy")
    assert result.returncode == 0
    assert "chown nobody:nogroup /app/legacy" in calls
    assert "deprecat" in result.stderr.lower()
    assert "VIDEO__VIDEO_OUTPUT_DIRECTORY" in result.stderr


def test_new_variable_wins_over_legacy(run_entrypoint):
    """When both are set, the new variable wins and no warning is printed."""
    result, calls = run_entrypoint(VIDEO__VIDEO_OUTPUT_DIRECTORY="/app/new", VIDEO_OUTPUT_DIRECTORY="/app/legacy")
    assert result.returncode == 0
    assert "chown nobody:nogroup /app/new" in calls
    assert not any("/app/legacy" in c for c in calls)
    assert "deprecat" not in result.stderr.lower()


def test_default_directory(run_entrypoint):
    """Without any variable the default /app/recordings is used."""
    result, calls = run_entrypoint()
    assert result.returncode == 0
    assert "chown nobody:nogroup /app/recordings" in calls
    assert "gosu nobody app-cmd" in calls


def test_empty_new_variable_falls_back(run_entrypoint):
    """An empty new variable is treated as unset."""
    result, calls = run_entrypoint(VIDEO__VIDEO_OUTPUT_DIRECTORY="")
    assert result.returncode == 0
    assert "chown nobody:nogroup /app/recordings" in calls


def test_path_traversal_rejected(run_entrypoint):
    """Paths containing '..' are rejected."""
    result, calls = run_entrypoint(VIDEO__VIDEO_OUTPUT_DIRECTORY="/app/../etc")
    assert result.returncode == 1
    assert calls == []


def test_path_outside_app_rejected(run_entrypoint):
    """Paths outside /app are rejected."""
    result, calls = run_entrypoint(VIDEO__VIDEO_OUTPUT_DIRECTORY="/data/vids")
    assert result.returncode == 1
    assert calls == []
