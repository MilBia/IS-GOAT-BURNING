from unittest.mock import AsyncMock

import pytest

from is_goat_burning.on_fire_actions.action_containers import OnceAction


@pytest.mark.asyncio
async def test_once_action_calls_action_only_once():
    """
    Tests that the OnceAction class only calls the provided action once.
    """
    # Create a mock action
    mock_action = AsyncMock()

    # Create an instance of the OnceAction class
    once_action = OnceAction(actions=[[mock_action, {}]])

    # Call the __call__ method twice
    await once_action()
    await once_action()

    # Assert that the action was only called once
    mock_action.assert_called_once()


@pytest.mark.asyncio
async def test_once_action_with_multiple_actions():
    """
    Tests that the OnceAction class can handle multiple actions.
    """
    # Create mock actions
    mock_action_1 = AsyncMock()
    mock_action_2 = AsyncMock()

    # Create an instance of the OnceAction class
    once_action = OnceAction(actions=[[mock_action_1, {}], [mock_action_2, {}]])

    # Call the __call__ method
    await once_action()

    # Assert that both actions were called
    mock_action_1.assert_called_once()
    mock_action_2.assert_called_once()
