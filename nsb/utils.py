"""Miscellaneous utility functions"""
import hashlib
import inspect
import typing as t
import warnings

import dill

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


def disable_warnings(disable: bool=True):
    r"""Disable all warnings"""
    if disable:
        warnings.filterwarnings("ignore")
    else:
        warnings.filterwarnings("default")

F = t.TypeVar("F", bound=t.Callable)
def _enforce_typehints_func(func: F) -> F:
    r"""
    Decorator to enforce typehints on a function

    This decorator will raise a TypeError if the typehints of the function
    are not satisfied.
    """
    def inner(*args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if not isinstance(value, sig.parameters[name].annotation):
                raise TypeError(f"Expected {sig.parameters[name].annotation} for {name}, got {type(value)}")
        return func(*args, **kwargs)
    return inner


C = t.TypeVar("C", bound=t.Type[type])
def _enforce_typehints_class(cls: C) -> C:
    r"""
    Decorator to enforce typehints on a class

    This decorator will raise a TypeError if the typehints of the class
    are not satisfied.
    """
    original_init = cls.__init__
    sig = inspect.signature(original_init)
    types = t.get_type_hints(original_init)
    def new_init(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if sig.parameters[name].annotation == inspect._empty:
                continue
            annotation_type = types[name]
            if not isinstance(value, annotation_type):
                raise TypeError(f"Expected {annotation_type} for {name}, got {type(value)}")
        print("Passed typehints check")
        print(f"Initialising with {args=}, {kwargs=}")
        original_init(self, *args, **kwargs)
    
    cls.__init__ = new_init
    return cls


def enforce_typehints(obj: C) -> C:
    r"""
    Decorator to enforce typehints on a function or class

    This decorator will raise a TypeError if the typehints of the function
    are not satisfied, and will enforce the typehints of the __init__ method
    of a class.
    """
    if inspect.isclass(obj):
        return _enforce_typehints_class(obj)
    return _enforce_typehints_func(obj)


def repeatable_hash(obj: t.Any) -> str:
    r"""
    Create a hash of an object that is repeatable

    This is useful for hashing objects that are not hashable, such as
    functions or classes.
    """
    return hashlib.md5(dill.dumps(obj)).hexdigest()