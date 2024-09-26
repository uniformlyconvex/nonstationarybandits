from __future__ import annotations

import abc
import builtins
import copy
import hashlib
import inspect
import typing as t
from pathlib import Path

import dill

import nsb.utils as utils
from nsb.agents.base import MABAgent, MABAgentParams
from nsb.environment import MABEnvironment, MABResult
from nsb.experiments.utils import repeat_runs


class Experiments:
    REGISTRY = []

    @staticmethod
    def register(experiment: t.Type[Experiment]) -> None:
        Experiments.REGISTRY.append(experiment)


Env = t.TypeVar("Env", bound=MABEnvironment)
Agent = t.TypeVar("Agent", bound=MABAgent)
Res = t.TypeVar("Res", bound=MABResult)
class Experiment(abc.ABC, t.Generic[Env, Agent, Res]):
    NO_RUNS = 50
    NO_TIMESTEPS = 1000

    def __init__(
        self,
        environment: Env,
        agents: list[Agent]
    ):
        self._environment = environment
        self._agents = agents

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        Experiments.register(cls)
    
    @property
    def environment(self) -> Env:
        return self._environment
    
    @property
    def agents(self) -> list[Agent]:
        return self._agents

    @property
    @abc.abstractmethod
    def filename(self) -> str:
        """Filename to save the experiment results"""

    def __hash__(self) -> int:
        """Hash the experiment to get a suffix for the filename"""
        # This is disgusting and I'm ashamed
        # We have to use hashlib because the hash() function is not stable;
        # it changes between runs (by design)
        classes = inspect.getmro(self.__class__)
        code = ''.join(
            inspect.getsource(cls)
            for cls in classes
            if cls not in vars(builtins).values()
        )
        return int(hashlib.sha256(code.encode()).hexdigest(), 16)

    def single_run(self, seed: int) -> dict[MABAgent, list[MABResult]]:
        utils.seed_random_states(seed=seed)
        
        # Copy the agents, since each process will have its own copy
        agents = copy.deepcopy(self._agents)
        environment = copy.deepcopy(self._environment)
        for agent in agents:
            agent.reset()
        
        results = {agent: [] for agent in agents}
        for t in range(self.NO_TIMESTEPS):
            for agent in agents:
                arm = agent.pick_action()
                result = environment.take_action(arm)
                agent.observe(result.observation)
                results[agent].append(result)
            environment.step()

        return results

    def run(
        self,
        no_runs: int | None=None,
        no_processes: int | None=None
    ) -> dict[MABAgent, list[list[MABResult]]]:
        no_runs = utils.defaulter(no_runs, self.NO_RUNS)
        return repeat_runs(self.single_run, no_runs, no_processes)

    def get_results(self) -> dict[MABAgent, list[list[MABResult]]]:
        # Traverse upwards to main.py
        curr_path = Path(__file__).resolve()
        for parent in curr_path.parents:
            if (parent / "main.py").exists():
                root = parent
                break
        else:
            raise FileNotFoundError("Could not find main.py")
        
        results_dir = root / "results"
        base_fname = self.filename + f'_{hash(self)}.pkl'
        filename = results_dir / base_fname
        folder = filename.parent
        folder.mkdir(parents=True, exist_ok=True)

        try:
            with open(filename, "rb") as f:
                return dill.load(f)
        except FileNotFoundError:
            results = self.run()
            with open(filename, "wb") as f:
                dill.dump(results, f)
            return results