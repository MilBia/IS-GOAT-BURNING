"""Unit tests for the OnceAction container.

These tests verify the core idempotency logic of the `OnceAction` class,
ensuring that wrapped actions are only executed once.
"""

from unittest.mock import AsyncMock

import pytest

from is_goat_burning.on_fire_actions.action_containers import OnceAction


@pytest.mark.asyncio
async def test_once_action_calls_action_only_once() -> None:
    """Verifies that a single wrapped action is only called on the first invocation."""
    mock_action = AsyncMock()
    once_action = OnceAction(actions=[(mock_action, {})])

    # Call the container twice
    await once_action()
    await once_action()

    # Assert that the underlying action was only called once
    mock_action.assert_called_once()


@pytest.mark.asyncio
async def test_once_action_with_multiple_actions() -> None:
    """Verifies that all wrapped actions are called on the first invocation."""
    mock_action_1 = AsyncMock()
    mock_action_2 = AsyncMock()
    once_action = OnceAction(actions=[(mock_action_1, {}), (mock_action_2, {})])

    await once_action()

    # Assert that both actions were called
    mock_action_1.assert_called_once()
    mock_action_2.assert_called_once()
