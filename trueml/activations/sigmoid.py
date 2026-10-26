from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Sigmoid:
  def __call__(self, x: npt.ArrayLike) -> npt.NDArray:
    return np.array(1 / (1 + np.exp(-x)))

  @staticmethod
  def grad(y_pred: npt.ArrayLike) -> npt.NDArray:
    return np.array(y_pred * (1 - y_pred))
