import numpy as np
from numpy.typing import ArrayLike


def min(x: ArrayLike) -> float:
    """
    Compute the minimum of the input array.

    min: minimum is the lowest numerical value in a dataset. It tells you the absolute lower boundary of your observations, reveals the minimum possible cost or measurement, and helps identify data limits or potential errors.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    float
        Minimum of the input array.

    Examples
    --------
    >>> min(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    1.0
    """
    return np.min(x)
