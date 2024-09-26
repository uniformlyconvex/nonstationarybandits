import typing as t
from dataclasses import dataclass

import numpy as np

import nsb.utils as utils
from nsb.agents.base import MABAgent, MABAgentParams
from nsb.type_hints import TimeVaryingParam


@dataclass(frozen=True, eq=True)
class EpsilonGreedyParams(MABAgentParams):
    epsilon: float | TimeVaryingParam[float]


class EpsilonGreedyAgent(
    MABAgent[EpsilonGreedyParams]
):
    def __str__(self) -> str:
        return f'ε-greedy(ε={"<callable" if callable(self._params.epsilon) else self._params.epsilon})'
    
    def __repr__(self) -> str:
        return self.__str__()

    def pick_action(self) -> int:
        epsilon = self._params.epsilon
        if callable(epsilon):
            epsilon = epsilon(self._t)
        if np.random.random() < epsilon:
            return np.random.randint(self._env.no_arms)
        means = [
            reward / count if count > 0 else 0.0
            for reward, count in zip(self._rewards, self._counts)
        ]
        return int(np.argmax(means))

    
    