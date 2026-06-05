from tinymlx.errors import absolute_error
import numpy as np


class MeanAbsoluteError:
    """
    Compute the mean absolute error (MAE).

    Formula:
        mean(|y_true - y_pred|)

    MAE measures the average magnitude of prediction errors without
    considering their direction. Unlike MSE, it is less sensitive to
    large outliers.

    Examples
    --------
    >>> mae = MeanAbsoluteError()
    >>> mae([10, 20], [8, 25])
    3.5
    """
    @staticmethod
    def surface(y_true, y_pred):
        return absolute_error(y_true, y_pred)

    def __call__(self, y_true, y_pred):
        return np.mean(absolute_error(y_true, y_pred))

    def grad(self, y_true, y_pred):
        """
        Compute gradient of MAE w.r.t predictions (dL/dy_pred).

        Returns:
            array-like
                Gradient of MAE with respect to predictions.

        Note:
            The gradient is undefined at zero; here we use subgradient:
            sign(y_pred - y_true)
        """
        return np.sign(y_pred - y_true)