from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
