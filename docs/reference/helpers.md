# Helpers & Utilities

**Module:** `trueml.helpers`

Benchmarking, profiling, and data generation utilities to facilitate machine learning experiments and performance testing.

---

## `timeit`

```python
@trueml.helpers.timeit
def func(*args, **kwargs)
```

A decorator that measures and prints the execution time of a function. The number of execution iterations is dynamically controlled by the `I` environment variable.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `callable` | — | The function to be timed. |

**Environment Variable `I`:**
- `I >= 1` (default 10): Runs the function `I` times sequentially, printing the execution time (in seconds) for each run.
- `I = -1`: Runs the function in an infinite loop, continuously printing the execution time alongside the function's name.

**Example:**
```python
import os
import numpy as np
from trueml.helpers import timeit
from trueml.linalg import matmul

os.environ["I"] = "3"

@timeit
def benchmark():
    A = np.random.randn(50, 50)
    B = np.random.randn(50, 50)
    matmul(A, B)

benchmark()
# 0.0412
# 0.0401
# 0.0405
```

---

## `memprofile`

```python
@trueml.helpers.memprofile
def func(*args, **kwargs)
```

A decorator that profiles the memory allocations of a function using Python's built-in `tracemalloc` library.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `callable` | — | The function to be memory-profiled. |

**Output:**
Prints the top 10 memory-allocating lines of code executed during the function call.

**Example:**
```python
from trueml.helpers import memprofile
import numpy as np

@memprofile
def allocate():
    X = np.zeros((10000, 10000))
    return X

allocate()
# <tracemalloc statistics output>
```

---

## `generate`

```python
trueml.helpers.generate(lower: int = 1, upper: int = 100, size: tuple = (1, 1)) -> np.ndarray
```

Generates a random integer matrix drawn from a discrete uniform distribution.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lower` | `int` | `1` | The inclusive lower bound of the generated integers. |
| `upper` | `int` | `100` | The exclusive upper bound of the generated integers. |
| `size` | `tuple` | `(1, 1)` | The shape of the output matrix. |

**Returns:** 
A NumPy array of the specified `size` populated with random integers.

**Example:**
```python
from trueml.helpers import generate

X = generate(lower=0, upper=5, size=(2, 3))
print(X)
# [[1, 4, 0],
#  [2, 2, 3]]
```

## See Also
- [linalg](linalg.md) — Mathematical primitives that can be benchmarked using these tools.
