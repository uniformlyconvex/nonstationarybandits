from dataclasses import dataclass

import torch.distributions as dist

from nsb.agents.base import MABAgent, MABAgentParams

@dataclass(frozen=True, eq=True, unsafe_hash=True)
class TSParams(MABAgentParams):
    priors: list[dist.ExponentialFamily]


class TSAgent(MABAgent[TSParams]):
    def __str__(self) -> str:
        return f"TSAgent(priors={self._params.priors})"
    
    @staticmethod
    def _is_conjugate(
        prior: dist.ExponentialFamily,
        likelihood: dist.ExponentialFamily
    ) -> bool:
        match likelihood:
            case dist.Bernoulli() | dist.Binomial() | dist.NegativeBinomial() | dist.Geometric():
                return isinstance(prior, dist.Beta)
            case dist.Poisson():
                return isinstance(prior, dist.Gamma)
            case dist.Categorical() | dist.Multinomial():
                return isinstance(prior, dist.Dirichlet)
            case _:
                raise NotImplementedError(f"Conjugate prior for {likelihood} not implemented")
    
    @staticmethod
    def _compute_posterior(
        prior: dist.ExponentialFamily,
        likelihood: dist.ExponentialFamily,
        samples: list[float]
    ) -> dist.ExponentialFamily:
        # p(x | theta) = f(x)g(theta)e^{ <\phi(\theta), T(x)> }
        # T is suff statistics, phi(theta) is natural params

        # p(x | theta) = exp( <t(x), theta> - F(theta) + k(x) )
        #  = exp(<t(x),\theta)>) * exp(-F(\theta)) * exp(k(x))
        # so f(x) = exp(k(x)), g(\theta) = exp(-F(\theta))

        


    def posteriors(self) -> list[dist.ExponentialFamily]:
        dist.normal