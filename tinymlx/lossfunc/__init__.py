"""
Loss functions — error measurement and gradient computation.

Each loss provides two methods following a consistent interface:
    __call__(y, y_pred) -> error    — element-wise error vector
    grad(X, error) -> (dw, db)      — gradients w.r.t. model parameters

The grad() method expects the signed residual (y - y_pred), not the output
of __call__(). This is because __call__() returns the loss magnitude
(for logging/reporting), while grad() needs the signed error to compute
derivatives with the correct sign.

Available losses:
    signed_error(y, y_pred)  — raw signed deviation (standalone function)
    AbsoluteError            — L1 loss, outlier-robust, bounded gradient
    SquaredError             — L2 loss, smooth everywhere, magnitude-sensitive
"""

from .signed_error import signed_error
from .absolute_error import AbsoluteError
from .squared_error import SquaredError

__all__ = [
    "signed_error",
    "AbsoluteError",
    "SquaredError",
]
