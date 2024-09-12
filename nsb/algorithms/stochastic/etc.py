from dataclasses import dataclass

import numpy as np

from nsb.algorithms.stochastic.base import MABAgent, MABAgentParams

@dataclass(frozen=True, eq=True)
class ExploreThenCommitParams(MABAgentParams):
    no_times_per_arm: int


class ExploreThenCommitAgent(
    MABAgent[ExploreThenCommitParams]
):
    def __str__(self) -> str:
        return f"ExploreThenCommitAgent(no_times_per_arm={self.params.no_times_per_arm})"
    
    def pick_action(self) -> int:
        if self.t < self.env.no_arms * self.params.no_times_per_arm:
            return self.t % self.env.no_arms

        assert all(count == self.params.no_times_per_arm for count in self.counts)
        means = [
            reward / count if count > 0 else 0.0
            for reward, count in zip(self.rewards, self.counts)
        ]
        return int(np.argmax(means))