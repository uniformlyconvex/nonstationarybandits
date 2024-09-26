import math

import numpy as np
import torch
import torch.distributions as dist
from torch.distributions import constraints

class Delta(dist.Distribution):
    arg_constraints = {"loc": constraints.real}
    has_rsample = False

    def __init__(self, loc: float, validate_args=None):
        self.loc = torch.as_tensor(loc)
        super().__init__(validate_args=validate_args)

    @property
    def mean(self) -> float:
        return self.loc
    
    @property
    def stddev(self) -> float:
        return 0.0

    def sample(self, sample_shape=torch.Size()) -> float:
        with torch.no_grad():
            return self.loc * torch.ones(sample_shape)


class NormalInverseGamma(dist.Distribution):
    arg_constraints = {
        'mu': constraints.real,
        'lambda_': constraints.positive,
        'alpha': constraints.positive,
        'beta': constraints.positive
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, mu: float, lambda_: float, alpha: float, beta: float, validate_args=None):
        self.mu = torch.as_tensor(mu)
        self.lambda_ = torch.as_tensor(lambda_)
        self.alpha = torch.as_tensor(alpha)
        self.beta = torch.as_tensor(beta)
        super().__init__(validate_args=validate_args)

    @property
    def mean(self) -> tuple[float, float]:
        if self.alpha <= 1:
            raise ValueError("Mean is undefined when alpha <= 1")
        mean = (self.mu, self.beta / (self.alpha - 1))
        return tuple(x.item() for x in mean)
    
    @property
    def mode(self) -> tuple[float, float]:
        mode = (self.mu, self.beta / (self.alpha + 1 + 1/2))
        return tuple(x.item() for x in mode)
    
    @property
    def stddev(self) -> tuple[float, float]:
        return map(math.sqrt, self.variance)
    
    @property
    def variance(self) -> tuple[float, float]:
        if self.alpha <= 2:
            raise ValueError("Variance is undefined when alpha <= 2")
        var_x = self.beta / ((self.alpha - 1) * (self.lambda_))
        var_sigma2 = self.beta**2 / ((self.alpha - 1)**2 * (self.alpha - 2))
        return tuple(x.item() for x in (var_x, var_sigma2))
    
    def sample(self, sample_shape=torch.Size()) -> tuple[np.ndarray, np.ndarray]:
        no_samples = sample_shape[0] if sample_shape else 1
        sigma2 = dist.InverseGamma(self.alpha, self.beta).sample((no_samples,))
        variances = sigma2 / self.lambda_
        scales = variances.sqrt()

        x_dists = dist.Normal(self.mu.float(), scales)
        xs = x_dists.sample()
        return xs.numpy(), sigma2.numpy()