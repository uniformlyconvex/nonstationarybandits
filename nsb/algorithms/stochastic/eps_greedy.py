import typing as t
from dataclasses import dataclass

import numpy as np

from nsb.base import NSBAgent, NSBAgentParams
from nsb.algorithms.stochastic.base import MABAgent, MABAgentParams
from nsb.environments.stochastic import MABObservation, MABEnvironment
from nsb.type_hints import TimeVaryingParam


@dataclass(frozen=True, eq=True)
class EpsilonGreedyParams(MABAgentParams):
    epsilon: float | TimeVaryingParam[float]


class EpsilonGreedyAgent(
    MABAgent[EpsilonGreedyParams]
):
    def __str__(self) -> str:
        callable_epsilon = callable(self.params.epsilon)
        if callable_epsilon:
            return "EpsilonGreedyAgent(epsilon=<callable>)"
        
        return f"EpsilonGreedyAgent(epsilon={self.params.epsilon})"

    def pick_action(self) -> int:
        epsilon = self.params.epsilon
        if callable(epsilon):
            epsilon = epsilon(self.t)
        if np.random.random() < epsilon:
            return np.random.randint(self.env.no_arms)
        means = [
            reward / count if count > 0 else 0.0
            for reward, count in zip(self.rewards, self.counts)
        ]
        return int(np.argmax(means))

    
    