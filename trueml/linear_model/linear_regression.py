from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(self, X, y, lr=0.01):
        self.lr = lr
        self.X = X
        self.y = y

        n_features = self.X.shape[1]
        n_targets = self.y.shape[1]
        self.weights = np.random.random((n_features, n_targets))
        self.bias = np.zeros(n_targets)

    def forward(self):
        return self.X @ self.weights + self.bias

    def train(self, lossfn, epochs=1000):
        self.loss_fn = lossfn

        for _epoch in range(epochs):
            y_pred = self.forward()

            dloss = self.loss_fn.grad(self.y, y_pred)

            dw = self.X.T @ dloss
            db = dloss.sum(axis=0)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
