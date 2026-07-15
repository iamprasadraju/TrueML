import numpy as np
from numpy.typing import ArrayLike

from ..errors import residual_error


class MSEloss:
    """Mean Squared Error loss function.

    Computes the mean of squared residuals between true and predicted values.

    .. math::

        \\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2

    MSE penalizes larger prediction errors more heavily than absolute
    error because errors are squared, making it sensitive to outliers.

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.losses import MSEloss
    >>> mse = MSEloss()
    >>> mse(np.array([10, 20]), np.array([8, 25]))
    14.5
    """

    @staticmethod
    def surface(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        """Compute the element-wise squared errors (loss surface).

        Parameters
        ----------
        y_true : ArrayLike of shape (n_samples,)
            Ground truth target values.
        y_pred : ArrayLike of shape (n_samples,)
            Predicted values.

        Returns
        -------
        ArrayLike of shape (n_samples,)
            Per-sample squared errors.
        """
        return residual_error(y_true, y_pred) ** 2

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Compute the mean squared error.

        Parameters
        ----------
        y_true : ArrayLike of shape (n_samples,)
            Ground truth target values.
        y_pred : ArrayLike of shape (n_samples,)
            Predicted values.

        Returns
        -------
        float
            The scalar MSE loss value.
        """
        return np.mean(residual_error(y_true, y_pred) ** 2)

    def grad(self, y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        """Compute gradient of MSE with respect to predictions.

        .. math::

            \\frac{\\partial \\text{MSE}}{\\partial \\hat{y}} =
            \\frac{2}{n} (\\hat{y} - y)

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
        """
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)
