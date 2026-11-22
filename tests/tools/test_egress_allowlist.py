"""Tests for the outbound allow-list tool (#212)."""

import os
from pathlib import Path

import pytest

from is_goat_burning.config import Settings
from tools import egress_allowlist as ea

HOOK = "https://discord.com/api/webhooks/123/secret-token"
SERVICE_HOOK = "https://discordapp.com/api/webhooks/456/other-token?wait=true"


@pytest.fixture(autouse=True)
def _isolate_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removes exported settings the tool reads, so a developer's shell cannot change the results."""
    for name in list(os.environ):
        if name.upper() in ("SOURCE", "DETECTION_STRATEGY") or name.upper().startswith(("DISCORD__", "EMAIL__", "GEMINI__")):
            monkeypatch.delenv(name)


def _settings(**kwargs: object) -> Settings:
    """Builds settings isolated from any local ``.env`` / ``.env.tests``."""
    kwargs.setdefault("source", "https://www.youtube.com/watch?v=abc")
    return Settings(_env_file=None, **kwargs)


def _pairs(settings: Settings) -> list[tuple[str, int]]:
    return [(e.host, e.port) for e in ea.required_endpoints(settings)]


def test_youtube_source_needs_page_and_segment_domains() -> None:
    assert _pairs(_settings()) == [(".youtube.com", 443), (".googlevideo.com", 443)]


def test_short_youtube_link_also_needs_its_host() -> None:
    assert ("youtu.be", 443) in _pairs(_settings(source="https://youtu.be/abc"))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("https://cdn.example.org/live/index.m3u8?token=x", [("cdn.example.org", 443)]),
        ("http://camera.example.org:8080/stream", [("camera.example.org", 8080)]),
        ("rtsp://user:pass@cam.example.org/live", [("cam.example.org", 554)]),
        ("/videos/local.mp4", []),
        ("0", []),
    ],
)
def test_other_sources(source: str, expected: list[tuple[str, int]]) -> None:
    assert _pairs(_settings(source=source)) == expected


def test_discord_hooks_only_when_enabled_service_hooks_always() -> None:
    off = _settings(discord={"use_discord": False, "hooks": [HOOK], "service_hooks": [SERVICE_HOOK]})
    assert ("discord.com", 443) not in _pairs(off)
    assert ("discordapp.com", 443) in _pairs(off)
    on = _settings(discord={"use_discord": True, "hooks": [HOOK, HOOK.replace("123", "789")]})
    assert _pairs(on).count(("discord.com", 443)) == 1


def test_smtp_only_when_emails_enabled() -> None:
    email = {
        "sender": "a@example.org",
        "sender_password": "hunter2",
        "recipients": ["b@example.org"],
        "email_host": "SMTP.Example.org",
        "email_port": 587,
    }
    assert ("smtp.example.org", 587) not in _pairs(_settings(email=email))
    assert ("smtp.example.org", 587) in _pairs(_settings(email={**email, "use_emails": True}))


@pytest.mark.parametrize("strategy", ["gemini", "hybrid_cloud", "gemini_decision", "hybrid_decision"])
def test_cloud_strategies_need_gemini(strategy: str) -> None:
    assert (ea.GEMINI_HOST, 443) in _pairs(_settings(detection_strategy=strategy, gemini={"api_key": "secret-key"}))


def test_classic_does_not_need_gemini() -> None:
    assert (ea.GEMINI_HOST, 443) not in _pairs(_settings(gemini={"api_key": "secret-key"}))


def test_squid_allowlist_lists_https_and_comments_the_rest() -> None:
    settings = _settings(
        source="https://www.youtube.com/watch?v=abc",
        discord={"use_discord": True, "hooks": [HOOK]},
        email={
            "use_emails": True,
            "sender": "a@example.org",
            "sender_password": "hunter2",
            "recipients": ["b@example.org"],
            "email_host": "smtp.example.org",
            "email_port": 587,
        },
    )
    endpoints = ea.required_endpoints(settings) + [ea.Endpoint("www.youtube.com", 443, "duplicate")]
    lines = ea.squid_allowlist(endpoints).splitlines()
    entries = [line for line in lines if not line.startswith("#")]
    assert entries == [".youtube.com", ".googlevideo.com", "discord.com"]
    assert any(line.startswith("#") and "smtp.example.org:587" in line for line in lines)


def test_output_contains_no_secrets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_file = tmp_path / "goat.env"
    env_file.write_text(
        "\n".join(
            [
                "SOURCE=https://www.youtube.com/watch?v=abc",
                "DETECTION_STRATEGY=hybrid_decision",
                "GEMINI__API_KEY=secret-key",
                "DISCORD__USE_DISCORD=true",
                f'DISCORD__HOOKS=["{HOOK}"]',
                f'DISCORD__SERVICE_HOOKS=["{SERVICE_HOOK}"]',
                "EMAIL__USE_EMAILS=true",
                "EMAIL__SENDER=a@example.org",
                "EMAIL__SENDER_PASSWORD=hunter2",
                'EMAIL__RECIPIENTS=["b@example.org"]',
                "EMAIL__EMAIL_HOST=smtp.example.org",
                "EMAIL__EMAIL_PORT=587",
            ]
        )
    )
    for args in ([], ["--squid"]):
        assert ea.main(["--env-file", str(env_file), *args]) == 0
        out = capsys.readouterr().out
        assert "discord.com" in out and "smtp.example.org" in out and ea.GEMINI_HOST in out
        for secret in ("secret-token", "other-token", "webhooks", "hunter2", "secret-key", "a@example.org"):
            assert secret not in out
