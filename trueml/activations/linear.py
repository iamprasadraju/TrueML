from numpy.typing import ArrayLike


def linear(x: ArrayLike) -> ArrayLike:
    """
    Linear activation function (identity).

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    ArrayLike
        Output array (same as input).
    """
    return x
