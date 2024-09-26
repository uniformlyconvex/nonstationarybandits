import copy
from dataclasses import dataclass

import numpy as np
import torch.distributions as dist

from nsb.agents.base import MABAgent, MABAgentParams, MABObservation
from nsb.environment import MABEnvironment
from nsb.distributions.conjugacy import PriorPosterior, Likelihood

@dataclass(frozen=True)
class TSParams(MABAgentParams):
    # For now we only accept likelihoods with conjugate priors
    likelihoods: tuple[Likelihood]

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)


class TSAgent(MABAgent[TSParams]):
    def __init__(self, params: TSParams, environment: MABEnvironment) -> None:
        super().__init__(params, environment)
        self._posteriors: list[list[dist.Distribution]] = []
        self._store_posteriors()

    def __str__(self) -> str:
        priors_text = str([likelihood.distribution_type for likelihood in self._params.likelihoods])
        return f"TSAgent(priors={priors_text})"
    
    def __repr__(self) -> str:
        return "TSAgent"
    
    def _store_posteriors(self) -> None:
        posteriors = copy.deepcopy([
            likelihood.prior.distribution
            for likelihood in self._params.likelihoods
        ])
        self._posteriors.append(posteriors)
    
    def observe(self, observation: MABObservation) -> None:
        super().observe(observation)
        self._params.likelihoods[observation.arm_pulled].update_prior([observation.reward])
        self._store_posteriors()


    def pick_action(self) -> int:
        param_samples = [
            likelihood.sample_prior_distribution()
            for likelihood in self._params.likelihoods
        ]
        beliefs: list[dist.Distribution] = [
            likelihood.distribution(**params)
            for likelihood, params in zip(self._params.likelihoods, param_samples)
        ]
        means = [belief.mean.item() for belief in beliefs]
        return np.argmax(means)
    

@dataclass(frozen=True)
class TSTopTwoParams(MABAgentParams):
    likelihoods: tuple[Likelihood]
    beta: float  # probability of picking the best arm

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)
    

class TSTopTwoAgent(TSAgent):
    _params: TSTopTwoParams

    def __str__(self) -> str:
        priors_text = str([likelihood.distribution_type for likelihood in self._params.likelihoods])
        return f"TSTopTwoAgent(priors={priors_text}, beta={self._params.beta})"
    
    def __repr__(self) -> str:
        return "TSTopTwoAgent"
    
    def pick_action(self) -> int:
        param_samples = [
            likelihood.sample_prior_distribution()
            for likelihood in self._params.likelihoods
        ]
        beliefs: list[dist.Distribution] = [
            likelihood.distribution(**params)
            for likelihood, params in zip(self._params.likelihoods, param_samples)
        ]
        means = [belief.mean.item() for belief in beliefs]
        best_arm = np.argmax(means)
        if np.random.rand() < self._params.beta:
            return best_arm
        second_best_arm = np.argsort(means)[-2]
        return second_best_arm