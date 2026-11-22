"""Tests for the live-stream recorder of the fire-rule evaluation harness (issue #109)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import subprocess
from unittest.mock import AsyncMock

import numpy as np
import pytest

from is_goat_burning.stream import YouTubeStream
from tools.fire_rule_eval import record
from tools.fire_rule_eval.__main__ import main

RESOLVED = "https://rr1.googlevideo.example/index.m3u8?token=RESOLVEDSECRET"
SOURCE = "https://www.youtube.com/watch?v=GOAT&si=SOURCESECRET"


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record.time, "sleep", lambda _s: None)
    monkeypatch.setattr(record.shutil, "which", lambda _name: "/usr/bin/ffmpeg")


def _fake_ffmpeg(fail_calls: tuple[int, ...] = (), stderr: str = ""):
    """Returns a ``subprocess.run`` stand-in writing the output file unless the call number is in ``fail_calls``."""
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        if len(calls) in fail_calls:
            return subprocess.CompletedProcess(command, 1, "", stderr)
        Path(command[-1]).write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, "", "")

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "record",
        "--source",
        SOURCE,
        "--out",
        str(tmp_path / "videos"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--clip-seconds",
        "5",
        *extra,
    ]


def test_record_names_clip_and_appends_scan_style_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(YouTubeStream, "resolve_url", AsyncMock(return_value=RESOLVED))
    fake = _fake_ffmpeg()
    monkeypatch.setattr(record.subprocess, "run", fake)

    assert main(_args(tmp_path, "--count", "1")) == 0

    command = fake.calls[0]
    assert command[command.index("-i") + 1] == RESOLVED
    assert command[command.index("-t") + 1] == "5.0"
    assert command[command.index("-c") + 1] == "copy"
    clips = json.loads((tmp_path / "manifest.json").read_text())["clips"]
    assert len(clips) == 1
    video = Path(clips[0]["path"])
    assert re.fullmatch(r"goat_\d{8}T\d{6}Z\.mp4", video.name)
    assert video.parent == tmp_path / "videos" and video.exists()
    assert clips[0] == {"name": f"goat__{video.stem}", "path": str(video), "label": "neg", "group": "goat"}


def test_failed_clip_does_not_stop_the_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resolve = AsyncMock(side_effect=[ValueError("Failed to resolve stream URL."), RESOLVED, RESOLVED])
    monkeypatch.setattr(YouTubeStream, "resolve_url", resolve)
    fake = _fake_ffmpeg(fail_calls=(1,))
    monkeypatch.setattr(record.subprocess, "run", fake)
    stamps = iter(["20261201T000000Z", "20261201T010000Z", "20261201T020000Z"])
    monkeypatch.setattr(record, "datetime", type("D", (), {"now": staticmethod(lambda _tz: _Stamp(next(stamps)))}))

    assert main(_args(tmp_path, "--count", "3")) == 0  # resolve fails, then ffmpeg fails, then success

    assert resolve.await_count == 3
    assert len(fake.calls) == 2
    clips = json.loads((tmp_path / "manifest.json").read_text())["clips"]
    assert [c["name"] for c in clips] == ["goat__goat_20261201T020000Z"]
    assert not (tmp_path / "videos" / "goat_20261201T010000Z.mp4").exists()  # failed output removed


class _Stamp:
    def __init__(self, text: str) -> None:
        self._text = text

    def strftime(self, _fmt: str) -> str:
        return self._text


def test_ffmpeg_timeout_is_logged_and_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(YouTubeStream, "resolve_url", AsyncMock(return_value=RESOLVED))

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(record.subprocess, "run", run)
    assert main(_args(tmp_path, "--count", "2")) == 0
    assert not (tmp_path / "manifest.json").exists()
    assert list((tmp_path / "videos").iterdir()) == []


def test_ctrl_c_during_ffmpeg_removes_partial_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(YouTubeStream, "resolve_url", AsyncMock(return_value=RESOLVED))

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        Path(command[-1]).write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(record.subprocess, "run", run)
    assert main(_args(tmp_path)) == 0
    assert not (tmp_path / "manifest.json").exists()
    assert list((tmp_path / "videos").iterdir()) == []


def _fake_features(rows: int):
    return lambda *_a, **_k: {"t": np.arange(rows, dtype=np.float64), "area": np.ones(rows)}


def test_delete_video_only_after_feature_steps_succeed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(YouTubeStream, "resolve_url", AsyncMock(return_value=RESOLVED))
    monkeypatch.setattr(record.subprocess, "run", _fake_ffmpeg())
    monkeypatch.setattr(record, "iter_check_frames", lambda *_a: iter(()))
    monkeypatch.setattr(record, "iter_frames_30fps", lambda *_a: iter(()))
    monkeypatch.setattr(record, "extract_features", _fake_features(5))
    flicker_results = iter([0, 4])  # first clip: flicker decodes nothing; second clip: fine
    monkeypatch.setattr(record, "extract_flicker_features", lambda *_a, **_k: _fake_features(next(flicker_results))())
    stamps = iter(["20261201T000000Z", "20261201T010000Z"])
    monkeypatch.setattr(record, "datetime", type("D", (), {"now": staticmethod(lambda _tz: _Stamp(next(stamps)))}))
    feats, flick = tmp_path / "feats", tmp_path / "flick"

    args = _args(tmp_path, "--count", "2", "--features", str(feats), "--flicker", str(flick), "--delete-video")
    assert main(args) == 0

    videos = tmp_path / "videos"
    assert (videos / "goat_20261201T000000Z.mp4").exists()  # flicker failed: kept
    assert not (videos / "goat_20261201T010000Z.mp4").exists()  # both steps stored rows: deleted
    assert (feats / "goat__goat_20261201T010000Z.npz").exists()
    assert (flick / "goat__goat_20261201T010000Z.npz").exists()
    assert not (flick / "goat__goat_20261201T000000Z.npz").exists()


def test_delete_video_without_feature_dir_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resolve = AsyncMock(return_value=RESOLVED)
    monkeypatch.setattr(YouTubeStream, "resolve_url", resolve)
    assert main(_args(tmp_path, "--count", "1", "--delete-video")) == 2
    resolve.assert_not_awaited()


def test_urls_never_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runs the real ``resolve_url`` (yt-dlp mocked) so its own log lines are covered too."""
    stream_logger = logging.getLogger("Stream")
    monkeypatch.setattr(stream_logger, "propagate", True)  # the app logger does not propagate to caplog otherwise
    monkeypatch.setattr(YouTubeStream, "_get_stream_info", AsyncMock(return_value={"url": RESOLVED}))
    monkeypatch.setattr(record.subprocess, "run", _fake_ffmpeg(fail_calls=(1,), stderr=f"{RESOLVED}: Server returned 403"))
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="Stream")  # `.env.tests` sets CRITICAL, which would hide a leak

    assert main(_args(tmp_path, "--count", "2")) == 0

    output = caplog.text + "".join(capsys.readouterr())
    assert "ffmpeg failed" in output and "<url>" in output
    assert "recorded, added to" in output
    for secret in ("RESOLVEDSECRET", "SOURCESECRET", "googlevideo", "watch?v="):
        assert secret not in output
    assert stream_logger.level == logging.DEBUG  # the temporary quieting is restored


def test_extract_skips_deleted_video_and_keeps_features(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    feats = tmp_path / "feats"
    feats.mkdir()
    stored = feats / "goat__gone.npz"
    np.savez_compressed(stored, t=np.arange(3.0))
    manifest = tmp_path / "manifest.json"
    entry = {"name": "goat__gone", "path": str(tmp_path / "gone.mp4"), "label": "neg", "group": "goat"}
    manifest.write_text(json.dumps({"clips": [entry]}))

    assert main(["extract", "--manifest", str(manifest), "--out", str(feats)]) == 0
    assert main(["flicker", "--manifest", str(manifest), "--out", str(tmp_path / "flick"), "--jobs", "1"]) == 0

    assert "video missing, skipped" in capsys.readouterr().out
    with np.load(stored) as data:
        assert len(data["t"]) == 3


def test_unexpected_error_does_not_stop_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(YouTubeStream, "resolve_url", AsyncMock(return_value=RESOLVED))
    fake = _fake_ffmpeg()
    calls = iter([OSError(f"cannot run ffmpeg -i {RESOLVED}"), None])

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        error = next(calls)
        if error is not None:
            raise error
        return fake(command, **kwargs)

    monkeypatch.setattr(record.subprocess, "run", run)
    caplog.set_level(logging.INFO)

    assert main(_args(tmp_path, "--count", "2")) == 0

    assert len(json.loads((tmp_path / "manifest.json").read_text())["clips"]) == 1
    assert "attempt 1 failed (OSError)" in caplog.text
    assert "RESOLVEDSECRET" not in caplog.text


def test_sleep_is_interval_minus_elapsed_and_zero_on_overrun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record, "_one_clip", lambda _args: None)
    clock = iter([0.0, 100.0, 3600.0, 7300.0, 7300.0])  # start, after clip 1 (100 s), start, after clip 2 (3700 s), start
    monkeypatch.setattr(record.time, "monotonic", lambda: next(clock))
    sleeps: list[float] = []
    monkeypatch.setattr(record.time, "sleep", sleeps.append)

    assert main(_args(tmp_path, "--count", "3", "--interval-seconds", "3600")) == 0

    assert sleeps == [3500.0, 0.0]  # no sleep after the last clip
