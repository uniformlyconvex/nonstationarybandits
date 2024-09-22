from __future__ import annotations

import abc
import typing as t
from dataclasses import dataclass, fields

import numpy as np
import plotly.graph_objects as go

from nsb.agents.base import MABAgent, MABAgentParams
from nsb.environment import MABEnvironment, MABResult


@dataclass
class MABTrackerTraces:
    mean_random_regret: go.Scatter
    upper_std_random_regret: go.Scatter
    lower_std_random_regret: go.Scatter
    mean_pseudo_regret: go.Scatter
    upper_std_pseudo_regret: go.Scatter
    lower_std_pseudo_regret: go.Scatter

    def __iter__(self) -> t.Generator[go.Scatter, None, None]:
        """Allow to iterate over the traces"""
        for field in fields(self):
            yield getattr(self, field.name)


Agent = t.TypeVar("Agent", bound=MABAgent)
Env = t.TypeVar("Env", bound=MABEnvironment)
Res = t.TypeVar("Res", bound=MABResult)
class MABTracker(abc.ABC, t.Generic[Agent, Env, Res]):
    """
    Tracker for recording the results of a single agent's actions over multiple
    runs in one environment.
    """
    def __init__(
        self,
        agent: Agent,
        environment: Env
    ) -> None:
        self.agent = agent
        self.environment = environment

        self._stack: list[list[Res]] = []  # ith element is the ith run
        self._ready_for_new_run = True

    @property
    def results(self) -> list[list[Res]]:
        return self._stack
    
    @property
    def no_time_steps(self) -> int:
        return max(len(run) for run in self.results)
    
    def track(self, result: Res) -> None:
        """Add a result to the current run"""
        if self._ready_for_new_run:
            self._stack.append([])
            self._ready_for_new_run = False
        self._stack[-1].append(result)

    def start_new_run(self) -> None:
        """Start a new run"""
        self._ready_for_new_run = True

    def _cumulative_regrets(self, regret_type: str) -> np.ndarray:
        """
        Return the cumulative regrets for each run.
        The (i,t)th element is the regret of the ith run at time t.
        """
        return np.cumsum(
            np.array([
                [getattr(result, regret_type) for result in run]
                for run in self.results
            ]),
            axis=1
        )

    def _mean_pm_std(self, regret_type: str, std: float) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the mean, upper std, and lower std of the regrets.
        """
        regrets = self._cumulative_regrets(regret_type)
        mean = np.mean(regrets, axis=0)
        std_dev = np.std(regrets, axis=0)
        return mean, mean + std * std_dev, mean - std * std_dev

    @property
    def cumulative_random_regrets(self) -> np.ndarray:
        """
        Return the cumulative random regrets for each run.
        The (i,t)th element is the random regret of the ith run at time t.
        """
        return self._cumulative_regrets("random_regret")

    @property
    def cumulative_pseudo_regrets(self) -> np.ndarray:
        """
        Return the cumulative pseudo regrets for each run.
        The (i,t)th element is the pseudo regret of the ith run at time t.
        """
        return self._cumulative_regrets("pseudo_regret")

    @classmethod
    def from_results(cls, agent: Agent, environment: Env, results: list[list[Res]]) -> MABTracker:
        tracker = cls(agent, environment)
        tracker._stack = results
        return tracker
    
    def traces(self, std: float) -> MABTrackerTraces:
        """
        Return the traces for the tracker.
        """
        mrr, usrr, lsrr = self._mean_pm_std("random_regret", std=std)
        mpr, uspr, lspr = self._mean_pm_std("pseudo_regret", std=std)

        return MABTrackerTraces(
            mean_random_regret=go.Scatter(
                x=list(range(self.no_time_steps)),
                y=mrr,
                mode="lines",
                name="Mean Random Regret"
            ),
            upper_std_random_regret=go.Scatter(
                x=list(range(self.no_time_steps)),
                y=usrr,
                mode="lines",
                name=f"Mean + {std} * Std Random Regret"
            ),
            lower_std_random_regret=go.Scatter(
                x=list(range(self.no_time_steps)),
                y=lsrr,
                mode="lines",
                name=f"Mean - {std} * Std Random Regret"
            ),
            mean_pseudo_regret=go.Scatter(
                x=list(range(self.no_time_steps)),
                y=mpr,
                mode="lines",
                name="Mean Pseudo Regret"
            ),
            upper_std_pseudo_regret=go.Scatter(
                x=list(range(self.no_time_steps)),
                y=uspr,
                mode="lines",
                name=f"Mean + {std} * Std Pseudo Regret"
            ),
            lower_std_pseudo_regret=go.Scatter(
                x=list(range(self.no_time_steps)),
                y=lspr,
                mode="lines",
                name=f"Mean - {std} * Std Pseudo Regret"
            )
        )



    
    