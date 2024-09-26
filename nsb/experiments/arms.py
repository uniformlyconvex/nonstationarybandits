"""Some utility functions for creating arms"""

from nsb.distributions.nonstationary import Constants, SinePlusGaussianNoise, ParameterisedNSD
from nsb.distributions.stationary import Delta


def _rotate_list(lst: list, n: int) -> list:
    return lst[n:] + lst[:n]

def constant_swapping_arms(
    no_arms: int,
    arm_values: list[float],
    swap_every: int,
    max_timesteps: int
) -> tuple[Constants]:
    """
    Arms that rotate between the values repeatedly.
    """
    swap_times = list(range(0, max_timesteps, swap_every))
    repeats_required = (len(swap_times) + 1) // len(arm_values) + 1
    values = arm_values * repeats_required
    values = [
        _rotate_list(values, i)
        for i in range(no_arms)
    ]
    arms = (
        ParameterisedNSD(
            Delta,
            loc=Constants(change_points=swap_times, values=values[i])
        )
        for i in range(no_arms)
    )
    return arms

def sine_swapping_arms(
    no_arms: int,
    mean: float | list[float],
    std: float | list[float],
    amplitude: float | list[float],
    frequency: float | list[float],
    phase: float | list[float]
) -> tuple[SinePlusGaussianNoise]:
    """
    Arms that rotate between the values repeatedly.
    """
    listify = lambda x: x if isinstance(x, list) else [x] * no_arms

    mean, std, amplitude, frequency, phase = map(
        listify,
        (mean, std, amplitude, frequency, phase)
    )
    arms = (
        SinePlusGaussianNoise(
            amplitude=amplitude[i],
            frequency=frequency[i],
            delay=phase[i],
            mean=mean[i],
            std=std[i]
        )
        for i in range(no_arms)
    )
    return arms