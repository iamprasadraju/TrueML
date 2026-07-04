import numpy as np

class BCELoss:
    """
    Compute the binary cross-entropy (BCE) loss.

    Formula:
        -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    BCE measures the performance of a classification model whose output is a probability value between 0 and 1.

    Examples
    --------
    >>> bce = BCELoss()
    >>> bce([1, 0], [0.9, 0.1])
    0.10536051565782628
    """

    @staticmethod
    def surface(y_true, y_pred):
        # Clip to prevent log(0) causing NaN/inf values
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def __call__(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(self.surface(y_true, y_pred))

    def grad(self, y_true, y_pred):
        """
        Compute gradient of BCE w.r.t predictions.

        Returns:
            array-like
                Gradient of the loss with respect to predictions (dL/dy_pred)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Clip to prevent division by zero in gradients
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        n = y_true.shape[0]
        
        # Standard analytical derivative of BCE divided by n for the mean reduction
        return (1 / n) * ((y_pred - y_true) / (y_pred * (1 - y_pred)))
