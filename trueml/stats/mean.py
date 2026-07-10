import numpy as np
from numpy.typing import ArrayLike


def mean(x: ArrayLike) -> float:
    """
    Compute the mean of the input array.

    mean: The mean (or average) is a statistical measure of central tendency. It summarizes a dataset by providing a single "typical" middle value.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    float
        Mean of the input array.

    formula
    -------
    μ = Σx / n

    where μ is the mean, Σx is the sum of all elements, and n is the number of elements.

    Examples
    --------
    >>> mean(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    3.0
    """
    return np.mean(x)
