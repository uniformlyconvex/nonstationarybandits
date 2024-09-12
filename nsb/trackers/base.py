import abc
import typing as t

import plotly.graph_objects as go

from nsb.base import NSBAgent, NSBAgentParams, NSBEnvironment, NSBResult

Agent = t.TypeVar("Agent", bound=NSBAgent)
Params = t.TypeVar("Params", bound=NSBAgentParams)
Env = t.TypeVar("Env", bound=NSBEnvironment)
Res = t.TypeVar("Res", bound=NSBResult)
class NSBTracker(abc.ABC, t.Generic[Agent, Params, Env, Res]):
    """Tracker for recording results of agent actions"""
    def __init__(
        self, 
        agents: t.Iterable[Agent],
        agents_params: t.Iterable[Params],
        environment: Env
    ) -> None:
        self.agents = agents
        self.agents_params = agents_params
        self.environment = environment

        self._stack = {
            agent: []
            for agent in agents
        }

    @property
    def results(self) -> dict[Agent, list[Res]]:
        return self._stack

    @property
    def no_time_steps(self) -> int:
        """Return the number of time steps"""
        return len(self._stack[self.agents[0]])
    
    def track(self, agent: Agent, result: Res) -> None:
        """Record the result of an agent's action"""
        self._stack[agent].append(result)

    @abc.abstractmethod
    def regret_traces(self) -> dict[Agent, list[go.Scatter]]:
        """Return a list of traces to plot"""

    @abc.abstractmethod
    def arm_traces(self) -> dict[Agent, list[go.Scatter]]:
        """Return a list of traces to plot"""