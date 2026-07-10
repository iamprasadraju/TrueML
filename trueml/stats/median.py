import numpy as np
from numpy.typing import ArrayLike


def median(x: ArrayLike) -> float:
    """
    Compute the median of the input array.

    median: The median is the middle value in a dataset when the values are arranged in ascending or descending order. It is a measure of central tendency that is less affected by outliers than the mean.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    float
        Median of the input array.

    formula
    -------
    If n is odd: median = x[(n+1)/2]
    If n is even: median = (x[n/2] + x[n/2 + 1]) / 2

    x: array should be sorted
    n: number of elements in the array

    Examples
    --------
    >>> median(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    3.0
    """
    return np.median(x)
