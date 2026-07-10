import numpy as np
from numpy.typing import ArrayLike


def var(x: ArrayLike) -> float:
    """
    Compute the variance of the input array.

    var: variance measures how far the numbers in a data set are spread out from their average (mean)

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    float
        Variance of the input array.

    formula
    -------
    σ² = Σ(x - μ)² / n

    where σ² is the variance, μ is the mean, and n is the number of elements.

    Examples
    --------
    >>> variance(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    2.0
    """
    return np.var(x)
