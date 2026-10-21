from __future__ import annotations

import numpy as np

from trueml.history import History


class LinearModel:
  def __init__(self, n_features, n_targets, lr=0.01, history=True):
    self.lr = lr

    self.weights = np.random.random((n_features, n_targets))
    self.bias = np.zeros(n_targets)
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
    dl_dw = self.X_train.T @ dL_dy_pred
    dl_db = dL_dy_pred.sum(axis=0)

    self.weights -= self.lr * dl_dw
    self.bias -= self.lr * dl_db

  def predict(self, X_test):
    return X_test @ self.weights + self.bias
