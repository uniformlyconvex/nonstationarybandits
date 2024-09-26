from dataclasses import dataclass

import numpy as np

import nsb.utils as utils
from nsb.agents.base import MABAgent, MABAgentParams

@dataclass(frozen=True, eq=True)
class UCBParams(MABAgentParams):
    error_probability: float


class UCBAgent(MABAgent[UCBParams]):
    def __str__(self) -> str:
        return f"UCB(err_prob={self._params.error_probability})"

    def ucb(self, arm: int) -> float:
        if self._counts[arm] == 0:
            return np.inf
        return (
            self._rewards[arm] / self._counts[arm]
            + np.sqrt(
                2 * np.log(1 / self._params.error_probability) / self._counts[arm]
            )
        )
    
    def pick_action(self) -> int:
        arm = max(
            range(self._env.no_arms),
            key=lambda arm: self.ucb(arm)
        )
        return arm
    

@dataclass(frozen=True, eq=True)
class UCBLogarithmicParams(MABAgentParams):
    alpha: float


class UCBLogarithmicAgent(UCBAgent):
    _params: UCBLogarithmicParams
    
    def __str__(self) -> str:
        return f"UCBLogarithmic(alpha={self._params.alpha})"
    
    def ucb(self, arm: int, timestep: int | None=None) -> float:
        timestep = timestep or self._t
        if self._counts[arm] == 0:
            return np.inf
        return (
            self._rewards[arm] / self._counts[arm]
            + np.sqrt(
                self._params.alpha * np.log(timestep) / (2 * self._counts[arm])
            )
        )