import numpy as np
from numpy.typing import ArrayLike

from ..errors import absolute_error


class MAEloss:
    """Mean Absolute Error loss function.

    Computes the mean of absolute residuals between true and predicted values.

    .. math::

        \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|

    MAE measures the average magnitude of prediction errors without
    considering their direction. Unlike MSE, it is less sensitive to
    large outliers.

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.losses import MAEloss
    >>> mae = MAEloss()
    >>> mae(np.array([10, 20]), np.array([8, 25]))
    3.5
    """

    @staticmethod
    def surface(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        """Compute the element-wise absolute errors (loss surface).

        Parameters
        ----------
        y_true : ArrayLike of shape (n_samples,)
            Ground truth target values.
        y_pred : ArrayLike of shape (n_samples,)
            Predicted values.

        Returns
        -------
        ArrayLike of shape (n_samples,)
            Per-sample absolute errors.
        """
        return absolute_error(y_true, y_pred)

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Compute the mean absolute error.

        Parameters
        ----------
        y_true : ArrayLike of shape (n_samples,)
            Ground truth target values.
        y_pred : ArrayLike of shape (n_samples,)
            Predicted values.

        Returns
        -------
        float
            The scalar MAE loss value.
        """
        return np.mean(absolute_error(y_true, y_pred))

    def grad(self, y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        """Compute gradient of MAE with respect to predictions.

        The sub-gradient is used since the absolute value function is
        not differentiable at zero:

        .. math::

            \\frac{\\partial \\text{MAE}}{\\partial \\hat{y}_i} =
            \\text{sign}(\\hat{y}_i - y_i)

        Parameters
        ----------
        y_true : ArrayLike of shape (n_samples,)
            Ground truth target values.
        y_pred : ArrayLike of shape (n_samples,)
            Predicted values.

        Returns
        -------
        ArrayLike of shape (n_samples,)
            Gradient of the loss with respect to predictions
            (``dL/dy_pred``).

        Notes
        -----
        The gradient is undefined at zero; here we use the subgradient
        ``sign(y_pred - y_true)``.
        """
        return np.sign(y_pred - y_true)
