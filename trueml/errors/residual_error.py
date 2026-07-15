from numpy.typing import ArrayLike


def residual_error(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    """Compute the signed residual error between actual and predicted values.

    .. math::

        e = y_{\\text{true}} - y_{\\text{pred}}

    Positive residuals indicate underprediction
    (:math:`y_{\\text{true}} > y_{\\text{pred}}`).
    Negative residuals indicate overprediction.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        Ground truth target values.
    y_pred : ArrayLike of shape (n_samples,)
        Predicted values.

    Returns
    -------
    ArrayLike of shape (n_samples,)
        Element-wise signed residuals.

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.errors import residual_error
    >>> residual_error(np.array([10, 20]), np.array([8, 25]))
    array([ 2, -5])
    """
    return y_true - y_pred
