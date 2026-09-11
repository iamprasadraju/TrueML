from __future__ import annotations

import numpy as np


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def grad(y_pred):
        return y_pred * (1 - y_pred)
