"""Shared fixtures for the fire detection tests."""

from collections.abc import Generator

import pytest

from is_goat_burning.fire_detection.detectors import clear_edge_verifier_cache
from is_goat_burning.fire_detection.verifier_health import reset_verifier_status


@pytest.fixture(autouse=True)
def _clear_edge_verifier_cache() -> Generator[None]:
    """Clears the module-level edge verifier cache around every test so no state leaks between tests."""
    clear_edge_verifier_cache()
    yield
    clear_edge_verifier_cache()


@pytest.fixture(autouse=True)
def _reset_verifier_status() -> Generator[None]:
    """Clears the module-level verifier availability state around every test so no state leaks between tests."""
    reset_verifier_status()
    yield
    reset_verifier_status()
