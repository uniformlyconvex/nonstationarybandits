from dataclasses import dataclass

import numpy as np

from nsb.algorithms.stochastic.base import MABAgent, MABAgentParams

@dataclass(frozen=True, eq=True)
class UCBParams(MABAgentParams):
    error_probability: float


class UCBAgent(
    MABAgent[UCBParams]
):
    def __str__(self) -> str:
        return f"UCBAgent(error_probability={self.params.error_probability})"

    def ucb(self, arm: int) -> float:
        if self.counts[arm] == 0:
            return np.inf
        return (
            self.rewards[arm] / self.counts[arm]
            + np.sqrt(
                2 * np.log(1 / self.params.error_probability) / self.counts[arm]
            )
        )
    
    def pick_action(self) -> int:
        ucbs = [self.ucb(arm) for arm in range(self.env.no_arms)]
        return int(np.argmax(ucbs))