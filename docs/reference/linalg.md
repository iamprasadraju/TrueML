# Linear Algebra Primitives

**Module:** `trueml.linalg`

Low-level linear algebra primitives. These operations are extracted into a dedicated module to keep model and loss logic focused entirely on machine learning concepts. This establishes a clean abstraction boundary that allows swapping in optimized BLAS implementations without touching the mathematical protocols of the library.

---

## `matmul`

```python
trueml.linalg.matmul(matrixA: np.ndarray, matrixB: np.ndarray) -> np.ndarray
```

A pure-Python, triple-loop implementation of matrix multiplication.

**Formula:**
$$
C_{ij} = \sum_{k=1}^{cols_A} A_{ik} B_{kj}
$$

**Complexity:** $O(m \cdot k \cdot n)$ time, where $A$ is $m \times k$ and $B$ is $k \times n$.

**Raises:**
- `ValueError`: If the number of columns in $A$ does not match the number of rows in $B$.

**Algorithm:**
This function allocates an empty zero matrix $C$, iterates over the rows of $A$ and columns of $B$, and computes the dot product for each cell using a naive accumulator. It is strictly for educational and debugging purposes to demonstrate how matrix multiplication works without hidden native C/Fortran layers.

**Example:**
```python
import numpy as np
from trueml.linalg import matmul

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = matmul(A, B)
# [[19, 22], [43, 50]]
```

---

## `npmatmul`

```python
trueml.linalg.npmatmul(matrixA: np.ndarray, matrixB: np.ndarray) -> np.ndarray
```

A NumPy-native wrapper for matrix multiplication.

**Implementation:**
Delegates directly to NumPy's `@` operator (`matrixA @ matrixB`), which internally hooks into highly optimized BLAS/LAPACK routines (e.g., OpenBLAS, MKL) written in C/Fortran.

---

## Comparison

| Property | `matmul` | `npmatmul` |
|----------|----------|------------|
| **Implementation** | Pure Python (triple nested loops) | NumPy `@` (BLAS) |
| **Readability** | Textbook algorithm, transparent | One-line mathematical operator |
| **Performance** | $O(n^3)$ interpreted (very slow) | Optimized C/Fortran (very fast) |
| **Use case** | Learning / algorithmic debugging | Production / large dataset training |

## See Also
- [helpers](helpers.md) — Benchmarking utilities like `@timeit` and `@memprofile` for testing `matmul` vs `npmatmul`.
