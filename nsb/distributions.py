"""
Some helper classes/functions for working with distributions. 

While most algorithms work just with numpy arrays, we use torch Distribution
objects here because numpy's random sampling methods generally don't provide a
`mean()` method, which we need. Additionally any gradient-based methods would
need tensors anyway.
"""

from __future__ import annotations

import abc
import typing as t
from dataclasses import dataclass, fields

import plotly.graph_objects as go
import torch.distributions as dist

import nsb.utils as utils
from nsb.type_hints import DistFn, TimeVaryingParam


@dataclass
class Traces:
    mean: go.Scatter
    upper_std: go.Scatter
    lower_std: go.Scatter
    samples: go.Scatter

    def __iter__(self) -> t.Generator[go.Scatter, None, None]:
        """Allow to iterate over the traces"""
        for field in fields(self):
            yield getattr(self, field.name)


class NSDist(abc.ABC):
    r"""
    NSDist is the base class for non-stationary distributions.
    """
    DEFAULT_TIMESTEPS: int=1_000
    DEFAULT_STD: float=2.0
    DEFAULT_SAMPLES: int=10

    def __init__(self, dist_fn: DistFn, name: str='') -> None:
        self._dist_fn = dist_fn
        self.name = name

    @property
    def dist_fn(self) -> DistFn:
        return self._dist_fn

    def get_dist(self, timestep: int) -> dist.Distribution:
        """Return the Distribution at the given timestep"""
        return self._dist_fn(timestep)

    def mean_trace(self, timesteps: t.Iterable[int]) -> go.Scatter:
        """Produces a Line Trace of the mean of the distribution over time"""
        mean_y_vals = [
            self.get_dist(timestep=time).mean.item()
            for time in timesteps
        ]
        trace = go.Scatter(
            x=list(timesteps),
            y=mean_y_vals,
            mode="lines",
            name="Mean"
        )
        return trace
    
    def std_bound_trace(self, timesteps: t.Iterable[int], std: float) -> go.Scatter:
        r"""
        Produces a Line Trace of the mean plus `std` times the standard
        deviation over time
        """
        values = []
        for time in timesteps:
            dist = self.get_dist(timestep=time)
            values.append((dist.mean + std * dist.stddev).item())

        trace = go.Scatter(
            x=list(timesteps),
            y=values,
            mode="lines",
            name=f"Mean {'+' if std >= 0 else '-'} {abs(std)} * StdDev"
        )
        return trace
    
    def samples_trace(self, timesteps: t.Iterable[int], no_samples: int) -> go.Scatter:
        r"""
        Produces a Scatter Trace of samples from the distribution over time
        """
        values = [
            self.get_dist(timestep=time).sample((no_samples,)).numpy()
            for time in timesteps
        ]
        x_vals, y_vals = utils.flatten_values(timesteps, values)
        trace = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name=f"Samples"
        )
        return trace
    
    def traces(
        self,
        timesteps: t.Iterable[int] | None=None,
        std: float | None=None,
        no_samples: int | None=None
    ) -> Traces:
        timesteps = utils.defaulter(timesteps, range(NSDist.DEFAULT_TIMESTEPS))
        std = utils.defaulter(std, NSDist.DEFAULT_STD)
        no_samples = utils.defaulter(no_samples, NSDist.DEFAULT_SAMPLES)
        
        return Traces(
            self.mean_trace(timesteps),
            self.std_bound_trace(timesteps, std),
            self.std_bound_trace(timesteps, -std),
            self.samples_trace(timesteps, no_samples)
        )


class ParameterisedNSD(NSDist):
    """
    A NSD where the distribution is parameterised by some parameters.

    The parameters must be passed as a dictionary to the constructor.
    """
    def __init__(
        self,
        dist: t.Type[dist.Distribution],
        name='',
        **kwargs: dict[str, TimeVaryingParam[float]]
    ) -> None:
        self._dist_type = dist
        dist_fn: DistFn = lambda t: dist(**{
            kwarg_name: kwarg_value(t) 
            for kwarg_name, kwarg_value in kwargs.items()
        })
        super().__init__(dist_fn, name)

    def __repr__(self) -> str:
        # Build a repr based on the class name and the name of the distribution
        torch_dist_name = self._dist_type.__name__
        return (
            f"Parameterised{torch_dist_name}" +
            (f" ({self.name})" if self.name else "")
        )