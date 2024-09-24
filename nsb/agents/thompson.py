from dataclasses import dataclass

import numpy as np
import torch.distributions as dist

from nsb.agents.base import MABAgent, MABAgentParams, MABObservation
from nsb.distributions.conjugacy import PriorPosterior, Likelihood

@dataclass(frozen=True)
class TSParams(MABAgentParams):
    # For now we only accept likelihoods with conjugate priors
    likelihoods: tuple[Likelihood]

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)


class TSAgent(MABAgent[TSParams]):
    def __str__(self) -> str:
        priors_text = str([likelihood.distribution_type for likelihood in self._params.likelihoods])
        return f"TSAgent(priors={priors_text})"
    
    def observe(self, observation: MABObservation) -> None:
        super().observe(observation)
        self._params.likelihoods[observation.arm_pulled].update_prior([observation.reward])

    def pick_action(self) -> int:
        param_samples = [
            likelihood.sample_prior_distribution()
            for likelihood in self._params.likelihoods
        ]
        beliefs: list[dist.Distribution] = [
            likelihood.distribution(**params)
            for likelihood, params in zip(self._params.likelihoods, param_samples)
        ]
        print(f"{beliefs=}")
        means = [belief.mean.item() for belief in beliefs]
        return np.argmax(means)