from numpy.typing import ArrayLike


def absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    """Compute the absolute error between actual and predicted values.

    .. math::

        e = |y_{\\text{true}} - y_{\\text{pred}}|

    Absolute error measures the magnitude of prediction error without
    considering its direction. The result is strictly non-negative.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        Ground truth target values.
    y_pred : ArrayLike of shape (n_samples,)
        Predicted values.

    Returns
    -------
    ArrayLike of shape (n_samples,)
        Element-wise absolute errors.

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.errors import absolute_error
    >>> absolute_error(np.array([10, 20]), np.array([8, 25]))
    array([2, 5])
    """
    return abs(y_true - y_pred)
