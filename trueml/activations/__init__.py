"""Activations — element-wise nonlinearities for model forward passes.

Pure functions with no internal state. Used inside ``model.forward()``
to transform the linear predictor into probabilities, thresholds, etc.

Separating activations from models keeps each model file focused
on its prediction formula and avoids duplicating activation code
when adding new models.

Available Activations
~~~~~~~~~~~~~~~~~~~~~
- :func:`sigmoid` — maps to :math:`(0, 1)`
- :func:`linear` — identity (no transformation)
"""

from .linear import linear
from .sigmoid import sigmoid

__all__ = [
    "sigmoid",
    "linear",
]
