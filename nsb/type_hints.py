import typing as t

import torch.distributions as dist

T = t.TypeVar('T')
TimeVaryingParam = t.Callable[[int], T]

DistFn = TimeVaryingParam[dist.Distribution]