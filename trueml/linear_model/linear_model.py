from __future__ import annotations

import numpy as np

from ..history import History


class LinearModel:
    def __init__(self, lr=0.01, history=True):
        self.lr = lr
        self.weights = None
        self.bias = 0
        self.history = History() if history else None

    def forward(self, X):
        if self.weights is None:
            self.weights = np.random.random((X.shape[1], 1))
        self.X = X
        return X @ self.weights + self.bias

    def backward(self, dL_dy_pred):
        """
        ŷ = Xw + b

        dŷ/dw = X
        dŷ/db = 1

        Loss = L(y_true, ŷ)

        dL/dw = dL/dŷ · dŷ/dw
              = X.T @ dL/dŷ

        dL/db = sum(dL/dŷ)
        """

        # dw = ∂L/∂w
        # db = ∂L/∂b
        dw = self.X.T @ dL_dy_pred
        db = np.sum(dL_dy_pred)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
