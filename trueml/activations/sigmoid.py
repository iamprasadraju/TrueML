import numpy as np
from numpy.typing import ArrayLike


def sigmoid(x: ArrayLike) -> ArrayLike:
    """Sigmoid (logistic) activation function.

    Maps any real-valued input to the range :math:`(0, 1)`, making
    it useful for generating probabilities in binary classification.

    .. math::

        \\sigma(x) = \\frac{1}{1 + e^{-x}}

    Parameters
    ----------
    x : ArrayLike
        Input array of any shape.

    Returns
    -------
    ArrayLike
        Output array with the same shape as ``x``, with each element
        mapped to :math:`(0, 1)`.

    Notes
    -----
    The derivative of the sigmoid is:

    .. math::

        \\sigma'(x) = \\sigma(x)(1 - \\sigma(x))

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.activations import sigmoid
    >>> sigmoid(np.array([0, 1, -1]))
    array([0.5       , 0.73105858, 0.26894142])
    """
    return 1 / (1 + np.exp(-x))
