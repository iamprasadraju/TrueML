# helpers — Utilities

**Module:** `tinymlx.helpers`

Benchmarking and data generation utilities for experiments.

---

## timeit

```python
tinymlx.helpers.timeit(func: callable) -> callable
```

Decorator that times function execution. The number of iterations is controlled by the `I` environment variable.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `callable` | — | Function to time |

**Environment variable:** `I` — number of iterations (default `10`). Set to `-1` for infinite looping.

**Behavior:**

- `I` >= 1: runs `func` that many times, prints each execution time.
- `I == -1`: runs `func` indefinitely, printing execution time each iteration.

**Example:**

```python
import os
os.environ["I"] = "5"

from tinymlx.gemm import matmul
from tinymlx.helpers import timeit
import numpy as np

@timeit
def bench():
    A = np.random.randn(100, 100)
    B = np.random.randn(100, 100)
    matmul(A, B)

bench()
```

---

## generate

```python
tinymlx.helpers.generate(lower: int = 1, upper: int = 100, size: tuple = (1, 1)) -> np.ndarray
```

Generates a random integer matrix.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lower` | `int` | `1` | Minimum value (inclusive) |
| `upper` | `int` | `100` | Maximum value (exclusive) |
| `size` | `tuple` | `(1, 1)` | Shape of the output array |

**Returns:** `np.ndarray` of shape `size` with random integers in `[lower, upper)`.

**Example:**

```python
from tinymlx.helpers import generate

X = generate(lower=0, upper=10, size=(3, 4))
# X.shape == (3, 4), values in [0, 10)
```

---

## memprofile

```python
tinymlx.helpers.memprofile(func: callable) -> callable
```

Decorator that profiles memory allocations of a function using `tracemalloc`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `callable` | — | Function to profile |

**Output:** Prints the top 10 memory-allocating lines after execution.

**Requires:** Python's `tracemalloc` module (standard library).

**Example:**

```python
from tinymlx.helpers import memprofile
from tinymlx.gemm import matmul
import numpy as np

@memprofile
def bench():
    A = np.random.randn(500, 500)
    B = np.random.randn(500, 500)
    matmul(A, B)

bench()
```

---

## Notes

- `timeit` and `memprofile` are decorators. When applied, the wrapped function no longer returns a value (prints timing/profiling output instead).
- For comparing `matmul` vs `npmatmul` performance, apply `timeit` to both and compare the printed timings.

## Related references

- [gemm](./gemm.md) — the matrix operations these utilities benchmark
