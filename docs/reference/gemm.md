# gemm — Matrix Operations

**Module:** `tinymlx.gemm`

Primitive matrix multiplication routines. Provides both a pure-Python implementation for pedagogical transparency and a NumPy-native wrapper for performance.

---

## matmul

```python
tinymlx.gemm.matmul(matrixA: np.ndarray, matrixB: np.ndarray) -> np.ndarray
```

Pure-Python triple-loop matrix multiplication.

**Input:** $A \in \mathbb{R}^{m \times k}$ — first matrix.

**Input:** $B \in \mathbb{R}^{k \times n}$ — second matrix.

**Output:** $C \in \mathbb{R}^{m \times n}$ where $C_{ij} = \sum_{t=1}^{k} A_{it} B_{tj}$.

**Raises:** `ValueError` if `matrixA.shape[1] != matrixB.shape[0]`.

**Implementation:** Nested `for` loops over rows of $A$, columns of $B$, and the shared inner dimension $k$. This is the textbook $O(mkn)$ algorithm.

**Example:**

```python
import numpy as np
from tinymlx.gemm import matmul

A = np.array([[1, 0, 4, 3],
              [3, 2, 1, 8],
              [1, 9, 6, 0]])
B = np.array([[0, 1, 3],
              [3, 8, 9],
              [0, 8, 3],
              [8, 6, 3]])

C = matmul(A, B)
```

---

## npmatmul

```python
tinymlx.gemm.npmatmul(matrixA: np.ndarray, matrixB: np.ndarray) -> np.ndarray
```

NumPy-native matrix multiplication.

**Input:** $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$.

**Output:** $C \in \mathbb{R}^{m \times n}$.

**Implementation:** Delegates to `matrixA @ matrixB` (NumPy's BLAS-optimized `__matmul__`).

**Example:**

```python
from tinymlx.gemm import npmatmul

C = npmatmul(A, B)
```

---

## Comparison

| Property | `matmul` | `npmatmul` |
|----------|----------|------------|
| Implementation | Pure Python (triple loop) | NumPy `@` (BLAS) |
| Readability | Textbook algorithm | One-line call |
| Performance | $O(n^3)$ interpreted | Optimized C/Fortran |
| Use case | Learning / debugging | Production / large data |

## Related references

- [helpers](./helpers.md) — benchmarking utilities (`timeit`, `memprofile`) for comparing `matmul` vs `npmatmul` performance
