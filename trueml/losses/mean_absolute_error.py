from __future__ import annotations

import numpy as np
import numpy.typing as npt


class MAELoss:
  def __call__(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray: return np.mean(np.abs(y_true - y_pred))

  @staticmethod
  def grad(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    n = y_true.size
    dL_dy_pred = (1 / n) * np.sign(y_pred - y_true)
    return np.array(dL_dy_pred)
