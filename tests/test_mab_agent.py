"""
Test functionality of base MAB agent.

Just checks that the rewards are tracked correctly.
"""
import typing as t

import pytest
import unittest.mock as mock

from nsb.environment import MABEnvironment, MABObservation
from nsb.agents.base import MABAgent, MABAgentParams

NO_ARMS = 5

@pytest.fixture
def agent() -> t.Type[MABAgent]:
    # To test ABCs, we need to make concrete implementations
    class ConcreteParams(MABAgentParams):
        def __hash__(self) -> int:
            return 0
        
    class ConcreteAgent(MABAgent[ConcreteParams]):
        def __str__(self) -> str:
            return 'ConcreteAgent'

        def pick_action(self) -> int:
            return 0

    class DummyEnvironment(MABEnvironment):
        def __init__(self):
            super().__init__([])

        @property
        def no_arms(self) -> int:
            return NO_ARMS

    env = DummyEnvironment()
    params = ConcreteParams()
    agent = ConcreteAgent(params, env)

    return agent


def test_setup(agent: MABAgent) -> None:
    # Check the agent set up its tracking okay
    assert agent.t == 0

    assert len(agent.rewards) == NO_ARMS
    assert len(agent.counts) == NO_ARMS

    assert all(reward == 0.0 for reward in agent.rewards)
    assert all(count == 0 for count in agent.counts)
