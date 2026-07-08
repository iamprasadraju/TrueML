import numpy as np
from numpy.typing import ArrayLike

from ..errors import residual_error


class MSEloss:
    """
    Compute the mean squared error (MSE).

    Formula:
        mean((y_true - y_pred)^2)

    MSE penalizes larger prediction errors more heavily than absolute error because errors are squared.

    Examples
    --------
    >>> mse = MeanSquaredError()
    >>> mse([10, 20], [8, 25])

    14.5
    """

    @staticmethod
    def surface(y_true: ArrayLike, y_pred: ArrayLike):
        return residual_error(y_true, y_pred) ** 2

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike):
        return np.mean(residual_error(y_true, y_pred) ** 2)

    def grad(self, y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        """
        Compute gradient of MSE w.r.t predictions.

        Returns:
            ArrayLike
                Gradient of the loss with respect to predictions (dL/dy_pred)
        """
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)
