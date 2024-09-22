import torch.distributions as dist

def is_conjugate(
    prior: dist.Distribution,
    likelihood: dist.Distribution
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
         

def compute_posterior(
    prior: dist.Distribution,
    likelihood: dist.Distribution,
    samples: list[float]
) -> dist.Distribution:
    match likelihood:
        case dist.Normal():
            like_prec = 1 / likelihood.variance