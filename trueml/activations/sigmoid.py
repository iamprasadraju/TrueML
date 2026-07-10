from numpy.typing import ArrayLike
import numpy as np


def sigmoid(x: ArrayLike) -> ArrayLike:
    """
    Sigmoid activation function.

    Formula:
        σ(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    ArrayLike
        Output array with sigmoid applied.

    Examples
    --------
    >>> sigmoid([0, 1, -1])
    array([0.5       , 0.73105858, 0.26894142])
    """
    return 1 / (1 + np.exp(-x))
