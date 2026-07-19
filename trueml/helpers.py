"""Utility functions for benchmarking and development.

This module provides decorators and helpers for timing, memory profiling,
and random data generation. These are intended for development and
benchmarking workflows, not for production training pipelines.
"""

import os
import time
import tracemalloc
from functools import wraps


def timeit(func):
    """Decorator that benchmarks a function's execution time.

    Runs the wrapped function ``I`` times (controlled by the ``I``
    environment variable, default ``10``) and prints the elapsed time
    for each iteration.

    Parameters
    ----------
    func : callable
        The function to benchmark.

    Returns
    -------
    callable
        A wrapper that runs ``func`` repeatedly and prints timing results.

    Notes
    -----
    Set the environment variable ``I=-1`` to run in an infinite loop
    (useful for sustained profiling).

    Examples
    --------
    >>> @timeit
    ... def my_function(x):
    ...     return x ** 2
    >>> my_function(42)  # prints timing for 10 iterations
    """
    def enc_func(*args):
        I = int(os.environ.get("I", 10))
        if I == -1:
            while 1:
                t = _timeit(func, *args)
                print(func.__name__, t)
        else:
            for _ in range(I):
                t = _timeit(func, *args)
                print(t)

    return enc_func


def _timeit(func, *args):
    """Run a single timed invocation of *func*.

    Parameters
    ----------
    func : callable
        The function to time.
    *args
        Positional arguments forwarded to *func*.

    Returns
    -------
    float
        Wall-clock elapsed time in seconds.
    """
    st = time.monotonic()
    func(*args)
    et = time.monotonic()

    return et - st  # in seconds


def generate(lower=1, upper=100, size=(1, 1)):
    """Generate a random integer matrix.

    Parameters
    ----------
    lower : int, default=1
        Lower bound (inclusive) for random values.
    upper : int, default=100
        Upper bound (exclusive) for random values.
    size : tuple[int, int], default=(1, 1)
        Shape of the output matrix as ``(rows, cols)``.

    Returns
    -------
    numpy.ndarray
        A random integer matrix of the specified shape.

    Examples
    --------
    >>> matrix = generate(lower=0, upper=10, size=(3, 4))
    >>> matrix.shape
    (3, 4)
    """
    import numpy as np

    matrix = np.random.randint(lower, upper, size)

    return matrix


def memprofile(func):
    """Decorator that profiles peak memory usage of a function.

    Uses :mod:`tracemalloc` to capture the top 10 memory-consuming
    lines during the execution of the wrapped function.

    Parameters
    ----------
    func : callable
        The function to profile.

    Returns
    -------
    callable
        A wrapper that prints memory statistics after execution.

    Examples
    --------
    >>> @memprofile
    ... def allocate():
    ...     return [0] * 1_000_000
    >>> allocate()  # prints top 10 memory-consuming lines
    """
    @wraps(func)
    def wrapper(*args):
        tracemalloc.start()
        _ = func(*args)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.stop()

        return _

    return wrapper
