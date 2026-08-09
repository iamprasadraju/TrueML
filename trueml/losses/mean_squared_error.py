from __future__ import annotations

import numpy as np

from ..plot import lossfn_plot


class MSELoss:
    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    @staticmethod
    def grad(y_true, y_pred):
        n = y_true.size
        dL_dy_pred = (-2 / n) * (y_true - y_pred)
        return dL_dy_pred

    def plot(self):
        lossfn_plot(self)
