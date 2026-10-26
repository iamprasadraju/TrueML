from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Linear:
  def __call__(self, x: npt.ArrayLike) -> npt.NDArray: return np.array(x)

  @staticmethod
  def grad(x: npt.ArrayLike) -> npt.NDArray:
      return np.ones_like(np.asarray(x), dtype=float)
    