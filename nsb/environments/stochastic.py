import typing as t
from dataclasses import dataclass

from nsb.base import NSBEnvironment, NSBObservation, NSBResult
from nsb.distributions import NSDist


@dataclass(frozen=True, eq=True)
class MABObservation(NSBObservation):
    """In this simple case, the observation is just the reward, plus the arm pulled."""
    arm_pulled: int
    reward: float


@dataclass(frozen=True, eq=True)
class MABResult(NSBResult):
    """
    The result of taking an action in the MAB environment.
    We record more than necessary for the tracker so we can decide later what
    to do with it.
    All regrets are instantaneous.
    """
    arm_pulled: int
    best_arm: int
    reward: float  # The reward drawn from the arm distribution
    expected_reward: float  # The expected reward of the arm distribution
    best_mean_reward: float  # The expected reward of the best arm

    @property
    def random_regret(self) -> float:
        return self.best_mean_reward - self.reward
    
    @property
    def pseudo_regret(self) -> float:
        return self.best_mean_reward - self.expected_reward

    @property
    def observation(self) -> MABObservation:
        return MABObservation(arm_pulled=self.arm_pulled, reward=self.reward)


class MABEnvironment(NSBEnvironment):
    def __init__(
        self,
        arms: t.Iterable[NSDist],
    ) -> None:
        self._arms = list(arms)
        self._t = 0

    @property
    def no_arms(self) -> int:
        return len(self._arms)

    def step(self) -> None:
        self._t += 1

    def take_action(self, action: int) -> MABResult:
        nsd = self._arms[action]
        dist = nsd.get_dist(self._t)

        reward = dist.sample().item()
        expected_reward = dist.mean.item()

        best_arm = max(
            range(self.no_arms),
            key=lambda arm: self._arms[arm].get_dist(self._t).mean.item()
        )
        best_arm_dist = self._arms[best_arm].get_dist(self._t)
        best_mean_reward = best_arm_dist.mean.item()

        return MABResult(
            arm_pulled=action,
            best_arm=best_arm,
            reward=reward,
            expected_reward=expected_reward,
            best_mean_reward=best_mean_reward
        )




