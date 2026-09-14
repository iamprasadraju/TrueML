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
- :class:`BCELoss` — Binary Cross Entropy
"""

from trueml.losses.binary_cross_entropy import BCELoss
from trueml.losses.mean_absolute_error import MAELoss
from trueml.losses.mean_squared_error import MSELoss

__all__ = [
    "BCELoss",
    "MAELoss",
    "MSELoss",
]
