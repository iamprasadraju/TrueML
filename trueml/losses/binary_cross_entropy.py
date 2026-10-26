from __future__ import annotations

import numpy as np
import numpy.typing as npt


class BCELoss:
  def __init__(self, eps: float = 1e-15): self.eps = eps
  def __call__(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    # np.clip to avoid log(0) = -inf. Keep probabilities away from 0 and 1
    y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

  @staticmethod
  def grad(y_true: npt.NDArray, y_pred: npt.NDArray, eps: float = 1e-15) -> npt.NDArray:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = y_true.size
    dL_dy_pred = (1 / n) * (y_pred - y_true) / (y_pred * (1 - y_pred))
    return np.array(dL_dy_pred)
