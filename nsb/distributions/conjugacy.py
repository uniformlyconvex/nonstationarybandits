from __future__ import annotations

import abc
import math
import typing as t

import torch.distributions as dist

import nsb.utils as utils
from nsb.distributions.stationary import NormalInverseGamma
from nsb.distributions.utils import IthElement, FunctionComposition


class _DistributionLikeMixin:
    def __init__(
        self,
        distribution: t.Type[dist.Distribution],
        **params: dict[str, float | PriorPosterior | tuple[t.Callable, PriorPosterior]],
    ) -> None:

        for key, value in params.items():
            # Create the attribute with the key and value
            setattr(self, f"_{key}", value)
        self._distribution = distribution
        self._distribution_params = tuple(params.keys())
        self._initial_param_values = tuple(params.values())

    def __str__(self) -> str:
        return f"{self.distribution_type.__name__}({self._distribution_params})"

    @property
    def distribution_type(self) -> t.Type[dist.Distribution]:
        return self._distribution

    def __hash__(self):
        params = [type(self)] + [
            (key, value)
            for key, value in zip(self._distribution_params, self._initial_param_values)
            if isinstance(value, float)
        ]
        return utils.repeatable_hash(tuple(params))


class Likelihood(_DistributionLikeMixin, abc.ABC):
    """Base class for likelihoods"""

    @abc.abstractmethod
    def update_prior(self, data: list[float]) -> None:
        """Update the prior distribution"""

    def distribution(self, **params: dict[str, float]) -> dist.Distribution:
        """Return the likelihood distribution"""
        return self._distribution(**params)

    def sample_prior_distribution(
        self, no_samples: int | None = None
    ) -> dict[str, float]:
        """Sample the distribution for params of the likelihood"""
        likelihood_params = {}
        for key in self._distribution_params:
            value = getattr(self, f"_{key}")
            # The value is either a float, a PriorPosterior, or a (callable, PriorPosterior) tuple
            if isinstance(value, PriorPosterior):
                value = value.sample_distribution(no_samples)
            elif isinstance(value, tuple):
                func: t.Callable = value[0]
                prior: PriorPosterior = value[1]
                value = func(prior.sample_distribution(no_samples))
            likelihood_params[key] = value
        return likelihood_params

    @property
    def prior(self) -> PriorPosterior:
        # For now there's only one param with a prior on it
        for key in self._distribution_params:
            value = getattr(self, f"_{key}")
            if isinstance(value, PriorPosterior):
                return value
        raise ValueError(f"Couldn't find the prior for {self}!")


class PriorPosterior(_DistributionLikeMixin):
    """Base class for prior-posterior distributions"""

    @property
    def distribution(self) -> dist.Distribution:
        dist_params: dict[str, float] = {}
        for key in self._distribution_params:
            value = getattr(self, f"_{key}")
            dist_params[key] = value
        dist = self._distribution(**dist_params)
        return dist

    def sample_distribution(self, no_samples: int | None = None) -> float:
        """Sample the distribution for params of the likelihood"""
        if no_samples is not None:
            return self.distribution.sample((no_samples,))
        return self.distribution.sample()

    def __hash__(self) -> int:
        params = [type(self)] + [
            (key, value)
            for key, value in self._initial_params.items()
            if isinstance(value, float)
        ]
        return utils.repeatable_hash(tuple(params))


class NormalPrior(PriorPosterior):
    _MIN_SCALE = 1e-32

    def __init__(self, loc: float, scale: float) -> None:
        super().__init__(distribution=dist.Normal, loc=loc, scale=scale)


# @utils.enforce_typehints
class NormalKnownScaleLikelihood(Likelihood):
    def __init__(self, scale: float, loc: NormalPrior) -> None:
        super().__init__(distribution=dist.Normal, scale=scale, loc=loc)

    def update_prior(self, data: list[float]) -> None:
        prior = self._loc
        mu_0 = prior._loc
        sigma_0_sq = prior._scale**2
        sigma_sq = self._scale**2
        n = len(data)
        sum_xi = sum(data)

        self._loc._loc = (1 / (1 / sigma_0_sq + n / sigma_sq)) * (
            mu_0 / sigma_0_sq + sum_xi / sigma_sq
        )
        self._loc._scale = max(
            1 / (1 / sigma_0_sq + n / sigma_sq), self._loc._MIN_SCALE
        )


class InverseGammaPrior(PriorPosterior):
    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__(distribution=dist.InverseGamma, alpha=alpha, beta=beta)


# @utils.enforce_typehints
class NormalKnownLocLikelihood(Likelihood):
    def __init__(self, loc: float, variance: InverseGammaPrior) -> None:
        super().__init__(distribution=dist.Normal, loc=loc, scale=(math.sqrt, variance))

    def update_prior(self, data: list[float]) -> None:
        mu = self.loc
        n = len(data)
        sum_xi_minus_mu_sq = sum((xi - mu) ** 2 for xi in data)

        self._variance._alpha = self._variance._alpha + n / 2
        self._variance._beta = self._variance._beta + sum_xi_minus_mu_sq / 2


class NormalInverseGammaPrior(PriorPosterior):
    def __init__(self, mu: float, lambda_: float, alpha: float, beta: float) -> None:
        super().__init__(
            distribution=NormalInverseGamma,
            mu=mu,
            lambda_=lambda_,
            alpha=alpha,
            beta=beta,
        )


# @utils.enforce_typehints
class NormalLikelihood(Likelihood):
    # Who knows, this one is a real mess
    def __init__(self, loc_variance: NormalInverseGammaPrior) -> None:
        self._loc_variance = loc_variance
        super().__init__(
            distribution=dist.Normal,
            loc=(IthElement(0), loc_variance),
            scale=(FunctionComposition((math.sqrt, IthElement(1))), loc_variance),
        )
        self._distribution_params = self._distribution_params + ("loc_variance",)

    @property
    def prior(self) -> NormalInverseGammaPrior:
        return self._loc_variance

    def sample_prior_distribution(self) -> dict[str, float]:
        loc_variance = self._loc_variance.sample_distribution()
        return {"loc": loc_variance[0].item(), "scale": loc_variance[1].item()}

    def update_prior(self, data: list[float]) -> None:
        mu_0 = self._loc_variance._mu
        nu = self._loc_variance._lambda_
        alpha = self._loc_variance._alpha
        beta = self._loc_variance._beta

        n = len(data)
        xbar = sum(data) / n

        new_mu = (nu * mu_0 + n * xbar) / (nu + n)
        new_lambda = nu + n
        new_alpha = alpha + n / 2
        new_beta = (
            beta
            + 0.5 * sum((xi - xbar) ** 2 for xi in data)
            + (n * nu * (xbar - mu_0) ** 2) / (2 * (nu + n))
        )
        self._loc_variance._mu = new_mu
        self._loc_variance._lambda_ = new_lambda
        self._loc_variance._alpha = new_alpha
        self._loc_variance._beta = new_beta
