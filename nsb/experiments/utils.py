import inspect
import pathos.multiprocessing as mp
import sys
import tokenize
import typing as t
from io import StringIO

import tqdm

from nsb.agents.base import MABAgent
from nsb.environment import MABResult


def repeat_runs(
    func: t.Callable[[int], dict[MABAgent, list[MABResult]]],
    no_runs: int,
    no_processes: int | None = None,
) -> dict[MABAgent, list[list[MABResult]]]:
    """
    Run the function `func` `no_runs` times in parallel using `no_cores` cores.
    """
    gettrace = getattr(sys, "gettrace", None)
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
                print(
                    f"Running {no_runs} runs in parallel using {no_processes} processes"
                )
                pool.map_async(func, range(no_runs), callback=lambda x: callback(x))
                pool.close()
                pool.join()

    agents = results[0].keys()
    return {agent: [result[agent] for result in results] for agent in agents}


def get_class_code_without_comments_and_docstrings(cls):
    # Get the source code of the class
    source_code = inspect.getsource(cls)

    # Use StringIO to treat the source code as a file for the tokenizer
    source_io = StringIO(source_code)

    # Tokenize the source code
    result = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(source_io.readline):
        token_type, token_string, (start_line, start_col), _, _ = tok

        # Skip comments (tokenize.COMMENT) and docstrings (tokenize.STRING)
        if token_type == tokenize.COMMENT or (
            token_type == tokenize.STRING and prev_toktype == tokenize.INDENT
        ):
            continue

        # Add non-comment, non-docstring tokens to the result
        if start_line > last_lineno:
            result.append("\n" * (start_line - last_lineno))
        elif start_col > last_col:
            result.append(" " * (start_col - last_col))

        result.append(token_string)
        prev_toktype = token_type
        last_lineno, last_col = start_line, start_col

    return "".join(result)


def get_agents() -> list[MABAgent]:
    # Get the current frame and the caller's frame
    current_frame = inspect.currentframe()
    try:
        caller_frame = current_frame.f_back  # Get the calling frame
        local_vars = caller_frame.f_locals  # Get local variables in the caller's frame

        # Filter variables that are instances of the target class
        instances = [var for var in local_vars.values() if isinstance(var, MABAgent)]
    finally:
        # Clean up frame references to avoid memory leaks
        del current_frame

    return instances
