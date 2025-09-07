import importlib

import pytest

from is_goat_burning import config  # Import the module itself


@pytest.fixture(autouse=True)
def mock_settings_environment(monkeypatch):
    """
    Mocks environment variables and reloads the settings module for every test.
    This ensures that tests use a consistent, mocked configuration.
    """
    # Set environment variables using monkeypatch
    monkeypatch.setenv("SOURCE", "test_source_from_conftest")
    monkeypatch.setenv("FIRE_DETECTION_THRESHOLD", "0.99")
    monkeypatch.setenv("CHECKS_PER_SECOND", "5")
    monkeypatch.setenv("LOGGING", "False")
    monkeypatch.setenv("VIDEO_OUTPUT", "False")
    monkeypatch.setenv("OPEN_CL", "False")
    monkeypatch.setenv("CUDA", "False")
    # ... set all your other desired test environment variables ...
    monkeypatch.setenv("EMAIL__USE_EMAILS", "False")
    monkeypatch.setenv("EMAIL__SENDER", "test@example.com")
    monkeypatch.setenv("EMAIL__SENDER_PASSWORD", "testpassword")
    monkeypatch.setenv("EMAIL__RECIPIENTS", '["recipient@example.com"]')
    monkeypatch.setenv("EMAIL__EMAIL_HOST", "smtp.test.com")
    monkeypatch.setenv("EMAIL__EMAIL_PORT", "1025")
    monkeypatch.setenv("DISCORD__USE_DISCORD", "False")
    monkeypatch.setenv("DISCORD__HOOKS", '["http://test.discord.hook"]')
    monkeypatch.setenv("VIDEO__SAVE_VIDEO_CHUNKS", "True")
    monkeypatch.setenv("VIDEO__VIDEO_OUTPUT_DIRECTORY", "/tmp/test-videos")
    monkeypatch.setenv("VIDEO__VIDEO_CHUNK_LENGTH_SECONDS", "5")
    monkeypatch.setenv("VIDEO__MAX_VIDEO_CHUNKS", "2")
    monkeypatch.setenv("VIDEO__CHUNKS_TO_KEEP_AFTER_FIRE", "2")

    # Reload the config module to apply the mocked env vars.
    # This is the crucial step that re-runs `settings = Settings()`.
    importlib.reload(config)
