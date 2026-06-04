import numpy as np

class SquaredError:
    def __call__(self, y, y_pred):
        return (y - y_pred) ** 2

    def grad(self, X, error):
        n = X.shape[0]
        grad_pred = -2 * error
        dw = (X.T @ grad_pred) / n
        db = np.mean(grad_pred)
        return dw, db
        