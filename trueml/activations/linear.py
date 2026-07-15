from numpy.typing import ArrayLike


def linear(x: ArrayLike) -> ArrayLike:
    """Linear (identity) activation function.

    Returns the input unchanged. This is the default activation for
    regression models.

    .. math::

        f(x) = x

    Parameters
    ----------
    x : ArrayLike
        Input array of any shape.

    Returns
    -------
    ArrayLike
        The input array, unmodified.

    Notes
    -----
    The derivative is trivially :math:`f'(x) = 1` everywhere.

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.activations import linear
    >>> linear(np.array([-5.0, 3.2]))
    array([-5. ,  3.2])
    """
    return x
