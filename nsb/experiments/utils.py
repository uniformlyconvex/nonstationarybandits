import pathos.multiprocessing as mp
import sys
import typing as t

import tqdm

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
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is not None and gettrace():
        print("Debugger is active, not using multiprocessing")
        results = [func(i) for i in range(no_runs)]
    else:
        # Make a tqdm progress bar, we'll update using callbacks
        results = []
        with mp.Pool(no_processes) as pool:
            with tqdm.tqdm(total=no_runs) as pbar:
                def callback(partial_results: list):
                    results.extend(partial_results)
                    pbar.update(len(partial_results))

                no_processes = pool._processes
                print(f"Running {no_runs} runs in parallel using {no_processes} processes")
                pool.map_async(func, range(no_runs), callback=lambda x: callback(x))
                pool.close()
                pool.join()
                



    agents = results[0].keys()
    return {
        agent: [result[agent] for result in results]
        for agent in agents
    }