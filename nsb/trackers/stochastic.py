import typing as t

import numpy as np
import plotly.graph_objects as go

from nsb.algorithms.stochastic.base import MABAgent, MABAgentParams
from nsb.environments.stochastic import MABObservation, MABResult, MABEnvironment
from nsb.trackers.base import NSBTracker


class MABTracker(
    NSBTracker[MABAgent, MABAgentParams, MABEnvironment, MABResult]
):
    def _scatter_collector(
        self,
        key: str,
        reductor: t.Callable[[list[float]], list[float] | np.ndarray],
        suffix: str | None=None,
        mode: str="lines"
    ) -> list[go.Scatter]:
        if suffix is None:
            suffix = ' '.join(key.split('_')).title()

        traces: dict[MABAgent, list[go.Scatter]] = {agent: [] for agent in self.agents}
        for agent in self.agents:
            trace = go.Scatter(
                x=list(range(self.no_time_steps)),
                y=reductor([
                    getattr(result, key)
                    for result in self.results[agent]
                ]),
                mode=mode,
                name=f"{str(agent)} {suffix or ''}"
            )
            traces[agent].append(trace)
        
        return traces
    
    def _cumulative_traces(self, key: str, suffix: str | None=None, mode: str="lines") -> dict[MABAgent, list[go.Scatter]]:
        reductor = lambda x: np.cumsum(x)
        return self._scatter_collector(key, reductor, suffix, mode)
    
    def _raw_traces(self, key: str, suffix: str | None=None, mode: str="lines") -> dict[MABAgent, list[go.Scatter]]:
        reductor = lambda x: x
        return self._scatter_collector(key, reductor, suffix, mode)
    
    def cumulative_random_regret_traces(self) -> dict[MABAgent, list[go.Scatter]]:
        return self._cumulative_traces("random_regret")
    
    def cumulative_pseudo_regret_traces(self) -> dict[MABAgent, list[go.Scatter]]:
        return self._cumulative_traces("pseudo_regret")
    
    def arm_traces(self) -> dict[MABAgent, list[go.Scatter]]:
        return self._raw_traces("arm_pulled", "Arm", mode="markers")
    
    def regret_traces(self) -> dict[MABAgent, list[go.Scatter]]:
        pseudo_traces = self.cumulative_pseudo_regret_traces()
        random_traces = self.cumulative_random_regret_traces()

        return {
            agent: pseudo_traces[agent] + random_traces[agent]
            for agent in self.agents
        }
