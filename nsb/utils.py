"""Miscellaneous utility functions"""

import typing as t

import numpy as np
import torch

def seed_random_states(seed: int) -> None:
    r"""
    Set the seeds of both torch and numpy.

    While it would be nice to just pass around a np.random.Generator object
    everywhere, we use torch's implementation of Distributions, which doesn't
    seem to support this, so we just set the seeds here.
    """
    np.random.seed(seed)  # Legacy method, may need changing at some point?
    torch.manual_seed(seed)


def clamp(value: float, min_value: float, max_value: float) -> float:
    r"""dConstrain a value to the interval [min_value, max_value]"""
    if value <= min_value:
        return min_value
    if value >= max_value:
        return max_value
    return value


def clamp_prob(value: float) -> float:
    r"""Constrain a value to the interval [0.0, 1.0]"""
    return clamp(value, 0.0, 1.0)


S = t.TypeVar("S")
T = t.TypeVar("T")
def defaulter(value: S | None, default: T) -> S | T:
    r"""Return default if value is not None"""
    return value if value is not None else default


def flatten_values(
    timesteps: list[int],
    y_values: list[np.ndarray]
) -> tuple[list[int], list[float]]:
    r"""Utility for working with plotly"""
    repeated_timesteps: list[int] = []
    flattened_values: list[float] = []

    for t, samples in zip(timesteps, y_values):
        for val in samples:
            repeated_timesteps.append(t)
            flattened_values.append(val)
    
    return repeated_timesteps, flattened_values
