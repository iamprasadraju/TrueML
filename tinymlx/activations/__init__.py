"""
Activations — element-wise nonlinearities for model forward passes.

Pure functions with no internal state. Used inside model.forward()
to transform the linear predictor into probabilities, thresholds, etc.

Separating activations from models keeps each model file focused
on its prediction formula and avoids duplicating activation code
when adding new models.

Available activations:
    sigmoid(x)    1 / (1 + exp(-x))          — maps to (0, 1)
    linear(x)     x                           — identity (no transformation)
    relu(x)       max(0, x)                   — rectified linear unit
    tanh(x)       (exp(x) - exp(-x)) / ...    — maps to (-1, 1)
    softmax(x)    exp(x) / sum(exp(x))        — multiclass probabilities
"""

from .sigmoid import sigmoid
from .linear import linear

__all__ = [
    "sigmoid",
    "linear",
]
