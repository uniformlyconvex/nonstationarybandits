import inspect
import typing as t
from dataclasses import dataclass
import numpy as np
import torch.distributions as dist

from nsb.distributions.nonstationary import NSDist


@dataclass(frozen=True, eq=True)
class MABObservation:
    """In this simple case, the observation is just the reward, plus the arm pulled."""
    arm_pulled: int
    reward: float


@dataclass(frozen=True, eq=True)
class MABResult:
    """
    The result of taking an action in the MAB environment.
    We record more than necessary for the tracker so we can decide later what
    to do with it.
    All regrets are instantaneous.
    """
    arm_pulled: int
    reward: float  # The reward drawn from the arm distribution
    expected_rewards: list[float]  # The expected rewards of the arm distributions

    @property
    def best_arm(self) -> int:
        """The arm with the highest expected reward."""
        return int(np.argmax(self.expected_rewards))
    
    @property
    def best_mean_reward(self) -> float:
        """The expected reward of the best arm."""
        return self.expected_rewards[self.best_arm]

    @property
    def random_regret(self) -> float:
        return self.best_mean_reward - self.reward
    
    @property
    def pseudo_regret(self) -> float:
        return self.best_mean_reward - self.expected_rewards[self.arm_pulled]

    @property
    def observation(self) -> MABObservation:
        return MABObservation(arm_pulled=self.arm_pulled, reward=self.reward)


def _new_getattribute(self, name: str) -> t.Any:
    r"""
    This is a hack to make it hard to access private attributes from the
    outside, i.e. agents can't `peek` into the environment by accident.
    
    __getattribute__ differs from __getattr__ in that it is called whenever
    any attribute is accessed, not just when it's missing.

    It won't stop someone who *really* wants to access the attribute, but
    it will make it harder to do so by accident. You could bypass this by
    calling `object.__getattribute__(env, "_private")` but that's a bit
    more deliberate than just `env._private`.
    """
    # Skip checking if the attribute name doesn't start with an underscore
    if not name.startswith("_"):
        return object.__getattribute__(self, name)
    
    # Otherwise, check if the call is internal
    stack = inspect.stack()
    for frame_info in stack[1:]:
        if frame_info.function == "__getattribute__":
            continue
        if frame_info.frame.f_locals.get("self") is self:
            # Call is internal
            return object.__getattribute__(self, name)
        
    # If we get here, the call is external and the attribute is private
    raise AttributeError(
        f"Cannot access attribute {name} from outside the class!"
    )

class MABEnvironment:
    def __init__(
        self,
        arms: t.Iterable[NSDist | dist.Distribution],
    ) -> None:
        # We turn Distributions into NSDists so we can use the same interface
        # for stationary and non-stationary distributions.
        self._arms = [
            arm if isinstance(arm, NSDist) else NSDist.from_dist(arm)
            for arm in arms
        ]
        self._t = 0

        self.__getattribute__ = _new_getattribute

    @property
    def arms(self) -> list[NSDist]:
        return self._arms

    def __getstate__(self) -> dict:
        """ Used for pickling instances """
        state = self.__dict__.copy()
        state['__getattribute__'] = object.__getattribute__
        return state

    def __setstate__(self, state: dict) -> None:
        """ Used for unpickling instances """
        state['__getattribute__'] = _new_getattribute
        self.__dict__.update(state)

    @property
    def no_arms(self) -> int:
        return len(self._arms)

    def step(self) -> None:
        self._t += 1

    def take_action(self, action: int) -> MABResult:
        dists = [arm.get_dist(self._t) for arm in self.arms]
        reward = dists[action].sample().item()

        expected_rewards = [
            dist.mean.item()
            for dist in dists
        ]

        return MABResult(
            arm_pulled=action,
            reward=reward,
            expected_rewards=expected_rewards
        )




