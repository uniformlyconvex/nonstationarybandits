"""
Abstract base classes to define interfaces.
"""

import abc
import inspect
import typing as t

class NSBObservation(abc.ABC):
    """
    Base class for observations. This should only be what the agent can observe;
    it shouldn't contain any information about the environment's internal state,
    including the regret.
    """
    pass


class NSBResult(abc.ABC):
    """
    Base class for action result. This should contain the reward and any forms
    of regret.
    """
    @property
    @abc.abstractmethod
    def observation(self) -> NSBObservation:
        """
        The observation from the environment for consumption by the agent.
        """

R = t.TypeVar("R", bound=NSBResult)
class NSBEnvironment(abc.ABC):
    """Base class for environments"""
    def __getattribute__(self, name: str) -> t.Any:
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
    
    @abc.abstractmethod
    def step(self) -> None:
        """
        Advance the environment by one timestep; that may involve mutating
        some internal state
        """

    @abc.abstractmethod
    def take_action(self, action: t.Any) -> R:
        r"""
        Generate a result based on the action taken.
        This should be for the tracker's consumption, though the agent observes
        the contained observation.
        """


class NSBAgentParams(abc.ABC):
    """
    Base class for agent parameters. Used to standardise instantiation of agents.
    """
    @abc.abstractmethod
    def __hash__(self) -> int:
        """
        Hash the parameters to identify agents in the tracker.
        """


Params = t.TypeVar("Params", bound=NSBAgentParams)
Env = t.TypeVar("Env", bound=NSBEnvironment)
Obs = t.TypeVar("Obs", bound=NSBObservation)
class NSBAgent(abc.ABC, t.Generic[Params, Env, Obs]):
    """Base class for agents"""
    @abc.abstractmethod
    def __init__(self, params: Params, environment: Env) -> None:
        """
        Initialise the agent with the given parameters and environment.

        The environment is passed in so the agent can observe any environment
        parameters needed to make decisions, such as the number of arms.
        It cannot access anything the environment doesn't expose as public
        attributes/methods, so the agent can't cheat by peeking into the
        environment's internal state.
        """
        self.params: Params = params
        self.env: Env = environment

    def __hash__(self) -> int:
        """
        Hash the agent based on its type and parameters. This is used to
        identify agents in the tracker.
        """
        return hash((type(self), self.params))
    
    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the agent. This is used for logging
        and debugging.
        """

    @abc.abstractmethod
    def pick_action(self) -> t.Any:
        """Choose an action based on the agent's internal state"""

    @abc.abstractmethod
    def observe(self, observation: Obs) -> None:
        """Update the agent's internal state based on the observation"""


