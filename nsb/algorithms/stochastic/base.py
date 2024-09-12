import abc
import typing as t
from dataclasses import dataclass

from nsb.base import NSBAgent, NSBAgentParams
from nsb.environments.stochastic import MABObservation, MABEnvironment


class MABAgentParams(NSBAgentParams):
    """
    A base class for parameters of agents in the multi-armed bandit setting.
    
    Still abstract as it has no __hash__ method.
    """
    pass


Params = t.TypeVar("Params", bound=MABAgentParams)
class MABAgent(
    NSBAgent[Params, MABEnvironment, MABObservation]
):
    """
    A base class for agents in the multi-armed bandit setting.
    
    Still abstract as it doesn't implement the `pick_action` method.
    """
    def __init__(self, params: Params, environment: MABEnvironment) -> None:
        super().__init__(params=params, environment=environment)
        self.rewards = [0.0 for _ in range(self.env.no_arms)]
        self.counts = [0 for _ in range(self.env.no_arms)]
        self.t = 0

    def observe(self, observation: MABObservation) -> None:
        self.rewards[observation.arm_pulled] += observation.reward
        self.counts[observation.arm_pulled] += 1
        self.t += 1