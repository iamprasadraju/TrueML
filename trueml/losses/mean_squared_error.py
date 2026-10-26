from __future__ import annotations

import numpy as np
import numpy.typing as npt


class MSELoss:
  def __call__(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    return np.mean(np.square(y_true - y_pred))

  @staticmethod
  def grad(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    n = y_true.size
    return np.array((2 / n) * (y_pred - y_true))
