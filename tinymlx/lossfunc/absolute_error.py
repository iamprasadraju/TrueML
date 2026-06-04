import numpy as np
# Absolute Error: |x - y| 

class AbsoluteError:
    def __call__(self, y_train, y_pred):
        # This returns the individual errors
        return np.abs(y_train -  y_pred)

    def grad(self, X, error):
        n = X.shape[0]
        grad_pred = -np.sign(error)
        dw = (X.T @ grad_pred) / n
        db = np.mean(grad_pred)
        return dw, db
        