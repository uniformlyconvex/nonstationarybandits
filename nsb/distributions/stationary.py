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
