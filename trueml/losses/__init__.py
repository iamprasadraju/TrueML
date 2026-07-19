"""
Loss functions for training — differentiable objectives.

Each loss class is callable and also exposes a :meth:`grad` method that
returns the analytical gradient with respect to predictions. Use losses
inside a training loop; for post-training evaluation, see
:mod:`trueml.metrics`.

Available Losses
~~~~~~~~~~~~~~~~
- :class:`MSEloss` — Mean Squared Error
- :class:`MAEloss` — Mean Absolute Error
"""

from .mean_absolute_error import MAEloss
from .mean_squared_error import MSEloss
from .binary_cross_entropy_loss import BCELoss
__all__ = [
    "MAEloss",
    "MSEloss",
    "BCELoss"
]
