from __future__ import annotations

import numpy as np

from trueml.history import History


class LinearRegression:
  def __init__(self, X, y, lr=0.01, history=False):
    self.lr = lr
    self.X = X
    self.y = y

    n_features = self.X.shape[1]
    n_targets = self.y.shape[1]
    self.weights = np.random.random((n_features, n_targets))
    self.bias = np.zeros(n_targets)

    self.history = History() if history else None

  def forward(self):
    return self.X @ self.weights + self.bias

  def training_step(self):
    if not hasattr(self, "loss_fn"):
      raise RuntimeError("Call train() before training steps.")

    y_pred = self.forward()

    loss = self.loss_fn(self.y, y_pred)
    dl_dy_pred = self.loss_fn.grad(self.y, y_pred)

    dl_dw = self.X.T @ dl_dy_pred
    dl_db = dl_dy_pred.sum(axis=0)

    # backward pass
    self.weights -= self.lr * dl_dw
    self.bias -= self.lr * dl_db
    return loss

  def train(self, loss_fn, epochs=1000):
    self.loss_fn = loss_fn

    for epoch in range(epochs):
      loss = self.training_step()
      self.history.append(epoch, loss=loss) if self.history is not None else None
