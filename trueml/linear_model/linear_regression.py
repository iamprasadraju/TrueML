from __future__ import annotations

import numpy as np

from trueml.helpers import bench_pbar


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

    def training_step(self):
        y_pred = self.forward()

        loss = self.loss_fn(self.y, y_pred)
        dl_dy_pred = self.loss_fn.grad(self.y, y_pred)

        dl_dw = self.X.T @ dl_dy_pred
        dl_db = dl_dy_pred.sum(axis=0)

        # backward pass
        self.weights -= self.lr * dl_dw
        self.bias -= self.lr * dl_db
        return loss

    @bench_pbar
    def train(self, loss_fn, epochs=1000):
        self.loss_fn = loss_fn
        for _epoch in range(epochs):
            self.training_step()


if __name__ == "__main__":
    from trueml.helpers import bench_pbar
    from trueml.losses import MSELoss

    X = np.random.random((10000, 100))
    y = np.random.random((10000, 1))

    model = LinearRegression(X, y)
    model.train(loss_fn=MSELoss(), epochs=10000)
