"""Linear algebra primitives.

Low-level matrix operations extracted into a dedicated module to keep
model and loss code focused on high-level logic. This is the layer that
can be replaced with optimized BLAS implementations later without
touching the rest of the library.

Available Operations
~~~~~~~~~~~~~~~~~~~~
- :func:`matmul` — naive :math:`O(n^3)` matrix multiply (educational)
- :func:`npmatmul` — NumPy-optimized multiply (``A @ B``)
"""

from .matmul import matmul, npmatmul

__all__ = [
    "matmul",
    "npmatmul",
]
