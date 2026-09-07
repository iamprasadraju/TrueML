from __future__ import annotations

import numpy as np

from trueml.history import History


class LinearModel:
    def __init__(self, n_features, lr=0.01, history=True):
        self.lr = lr

        self.weights = np.random.random((n_features, 1))
        self.bias = 0.0
        self.history = History() if history else None

    def forward(self, X_train):
        self.X_train = X_train
        return self.X_train @ self.weights + self.bias

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
        dw = self.X_train.T @ dL_dy_pred
        db = np.sum(dL_dy_pred)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def predict(self, X_test):
        return X_test @ self.weights + self.bias
