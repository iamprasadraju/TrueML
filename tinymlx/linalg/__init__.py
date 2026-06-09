"""
Linalg — low-level linear algebra primitives.

Extracted into a dedicated module to keep model and loss code focused
on high-level logic. This is the layer that can be replaced with
optimized BLAS implementations later without touching the rest of
the library.

Available operations:
    matmul(A, B)      — naive O(n³) matrix multiply (educational)
    npmatmul(A, B)    — numpy-optimized multiply (A @ B)
"""

from .matmul import matmul, npmatmul

__all__ = [
    "matmul",
    "npmatmul",
]
