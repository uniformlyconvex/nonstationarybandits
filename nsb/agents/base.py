from __future__ import annotations

import abc
import typing as t

from nsb.environment import MABObservation, MABEnvironment


class MABAgentParams:
    """
    A base class for parameters of agents in the multi-armed bandit setting.
    """


Params = t.TypeVar("Params", bound=MABAgentParams)
class MABAgent(abc.ABC, t.Generic[Params]):
    """
    A base class for agents in the multi-armed bandit setting.
    """
    def __init__(self, params: Params, environment: MABEnvironment) -> None:
        self._params: Params = params
        self._env: MABEnvironment = environment

        self.reset()

    def reset(self):
        self._rewards = [0.0 for _ in range(self._env.no_arms)]
        self._counts = [0 for _ in range(self._env.no_arms)]
        self._t = 0

    def __hash__(self) -> int:
        """
        Hash the agent based on its type and parameters. This is used to identify
        agents in trackers.
        """
        return hash((type(self), self._params))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def observe(self, observation: MABObservation) -> None:
        self._rewards[observation.arm_pulled] += observation.reward
        self._counts[observation.arm_pulled] += 1
        self._t += 1

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the agent. This is used for logging and
        debugging.
        """

    def __repr__(self) -> str:
        """
        Return a string representation of the agent. This is used for logging and
        debugging.
        """
        return self.__str__()

    @abc.abstractmethod
    def pick_action(self) -> int:
        """Choose an action based on internal state"""