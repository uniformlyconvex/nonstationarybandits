"""
Some helper classes/functions for working with distributions. 

While most algorithms work just with numpy arrays, we use torch Distribution
objects here because numpy's random sampling methods generally don't provide a
`mean()` method, which we need. Additionally any gradient-based methods would
need tensors anyway.
"""

from __future__ import annotations

import abc
import bisect
import typing as t
from dataclasses import dataclass, fields

import numpy as np
import plotly.graph_objects as go
import torch.distributions as dist

import nsb.utils as utils
from nsb.distributions.stationary import Delta
from nsb.type_hints import DistFn, TimeVaryingParam


@dataclass
class Traces:
    mean: go.Scatter
    upper_std: go.Scatter
    lower_std: go.Scatter
    samples: go.Scatter
    mean_to_time_t: go.Scatter

    def __iter__(self) -> t.Generator[go.Scatter, None, None]:
        """Allow to iterate over the traces"""
        for field in fields(self):
            yield getattr(self, field.name)


class _ReturnDist:
    def __init__(self, dist: dist.Distribution) -> None:
        self.dist = dist

    def __call__(self, time: int) -> dist.Distribution:
        return self.dist

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

    def mean_to_time_t_trace(self, timesteps: t.Iterable[int]) -> go.Scatter:
        """Produces a Line Trace of the mean of the distribution up to time t"""
        mean_y_vals = [
            self.get_dist(timestep=time).mean.item()
            for time in range(max(timesteps)+1)
        ]
        sums = np.cumsum(mean_y_vals)
        mean_to_time_t = [sums[time]/(time+1) for time in range(max(timesteps)+1)]
        trace = go.Scatter(
            x=list(range(max(timesteps)+1)),
            y=mean_to_time_t,
            mode="lines",
            name="Mean to time t"
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
            mean=self.mean_trace(timesteps),
            upper_std=self.std_bound_trace(timesteps, std),
            lower_std=self.std_bound_trace(timesteps, -std),
            samples=self.samples_trace(timesteps, no_samples),
            mean_to_time_t=self.mean_to_time_t_trace(timesteps)
        )
    
    @classmethod
    def from_dist(
        cls,
        dist: dist.Distribution,
        name: str=''
    ) -> NSDist:
        """Create a NSDist from a torch Distribution"""
        return cls(_ReturnDist(dist), name)


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
        self._params = kwargs

        super().__init__(self.dist_fn, name)
        
    def dist_fn(self, timestep: int) -> dist.Distribution:
        return self._dist_type(**{
            kwarg_name: kwarg_value(timestep)
            for kwarg_name, kwarg_value in self._params.items()
        })

    def __repr__(self) -> str:
        # Build a repr based on the class name and the name of the distribution
        torch_dist_name = self._dist_type.__name__
        return (
            f"Parameterised{torch_dist_name}" +
            (f" ({self.name})" if self.name else "")
        )
    

class Constants:
    """
    Creates a time-varying parameter where the value is a step function.
    """
    def __init__(self, change_points: list[int], values: list[int]):
        self.change_points = change_points
        self.values = values

    def __call__(self, time: int) -> float:
        idx = bisect.bisect_right(self.change_points, time)
        return self.values[idx]
    


class FunctionPlusGaussianNoise(ParameterisedNSD):
    def __init__(self, fn: TimeVaryingParam[float], std: TimeVaryingParam[float]):
        super().__init__(
            dist=dist.Normal,
            loc=fn,
            scale=std
        )


class SinePlusGaussianNoise(FunctionPlusGaussianNoise):
    def __init__(self, amplitude: float, frequency: float, delay: float, mean: float, std: float):
        self._amplitude = amplitude
        self._frequency = frequency
        self._delay = delay
        self._mean = mean
        self._std = std
        super().__init__(
            fn=self._mean_callable,
            std=self._std_callable
        )
        

    def _mean_callable(self, time: int) -> float:
        return self._mean + self._amplitude * np.sin(2*np.pi*(time - self._delay)/self._frequency)
    
    def _std_callable(self, time: int) -> float:
        return self._std