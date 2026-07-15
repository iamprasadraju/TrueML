"""Error functions — raw element-wise computation primitives.

Errors are raw, element-wise computations (e.g., :math:`y - \\hat{y}`).
They do **not** aggregate across the dataset and they do **not** define
gradients. Use the :mod:`trueml.losses` module for differentiable
objectives that participate in training.

Available Errors
~~~~~~~~~~~~~~~~
- :func:`residual_error` — signed difference :math:`y - \\hat{y}`
- :func:`absolute_error` — unsigned magnitude :math:`|y - \\hat{y}|`
"""

from .absolute_error import absolute_error
from .residual_error import residual_error

__all__ = [
    "absolute_error",
    "residual_error",
]
