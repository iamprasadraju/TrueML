from __future__ import annotations

import numpy as np


class MSELoss:
    def __init__(self, uops=False):
        self.uops = uops

    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def grad(self, y_true, y_pred):
        n = y_true.size
        return (2 / n) * (y_pred - y_true)
