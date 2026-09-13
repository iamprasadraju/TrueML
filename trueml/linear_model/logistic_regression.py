from __future__ import annotations

import numpy as np


class LogisticRegression:
    def __init__(self, X, y, lr=0.01, history=False):
        self.lr = lr
        self.X = X
        self.y = y

        n_features = self.X.shape[1]
        n_targets = self.y.shape[1]
        self.weights = np.random.random((n_features, n_targets))
        self.bias = np.zeros(n_targets)

    def forward(self):
        return self.X @ self.weights + self.bias

    def train(self, activ_fn, loss_fn, epochs=1000):
        self.loss_fn = loss_fn
        self.activ_fn = activ_fn

        for _epoch in range(epochs):
            logits = self.forward()
            y_pred = self.activ_fn(logits)

            dloss_fn = self.loss_fn.grad(self.y, y_pred)
            dactiv_fn = self.activ_fn.grad(y_pred)

            dz = dloss_fn * dactiv_fn

            dw = self.X.T @ dz
            db = dz.sum(axis=0)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
