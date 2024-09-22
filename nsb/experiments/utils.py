import multiprocessing as mp
import typing as t

from nsb.agents.base import MABAgent
from nsb.environment import MABResult


def repeat_runs(
    func: t.Callable[[int], dict[MABAgent, list[MABResult]]],
    no_runs: int,
    no_processes: int | None=None
) -> dict[MABAgent, list[list[MABResult]]]:
    """
    Run the function `func` `no_runs` times in parallel using `no_cores` cores.
    """
    with mp.Pool(no_processes) as pool:
        no_processes = pool._processes
        print(f"Running {no_runs} runs in parallel using {no_processes} processes")
        results = pool.map(func, range(no_runs))

    agents = results[0].keys()
    return {
        agent: [result[agent] for result in results]
        for agent in agents
    }