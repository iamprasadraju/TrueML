# Errors

**Module:** `trueml.errors`

The `errors` module contains raw computation primitives for measuring discrepancies between target values and predictions.

## Design Philosophy: Errors vs. Losses

In TrueML, errors are raw, element-wise computations (e.g., $y_i - \hat{y}_i$). They do not aggregate across the dataset and they do not define gradients. 

Losses (like `MSEloss` or `MAEloss`), on the other hand, aggregate these errors (e.g., via `np.mean()`) and mathematically define how the error translates into a gradient for model updating. The error functions live in their own module to keep the math clean and reusable.

---

## `residual_error`

```python
trueml.errors.residual_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Computes the raw, signed residual error between actual and predicted values.

**Formula:**
$$
e = y_{true} - y_{pred}
$$

**Properties:**
- Positive error indicates underprediction ($y_{true} > y_{pred}$).
- Negative error indicates overprediction ($y_{true} < y_{pred}$).
- Errors can cancel out if aggregated naively.

**Example:**
```python
from trueml.errors import residual_error
import numpy as np

y_true = np.array([10, 20])
y_pred = np.array([8, 25])

print(residual_error(y_true, y_pred))
# [2, -5]
```

---

## `absolute_error`

```python
trueml.errors.absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Computes the absolute magnitude of the error, discarding the sign.

**Formula:**
$$
e = |y_{true} - y_{pred}|
$$

**Properties:**
- Strictly non-negative.
- Accurately reflects the magnitude of the discrepancy without allowing positive and negative errors to cancel out.

**Example:**
```python
from trueml.errors import absolute_error
import numpy as np

y_true = np.array([10, 20])
y_pred = np.array([8, 25])

print(absolute_error(y_true, y_pred))
# [2, 5]
```

---

## Comparison Table

| Name | Formula | Range | Used By |
|------|---------|-------|---------|
| **residual_error** | $y - \hat{y}$ | $(-\infty, \infty)$ | `MSEloss` (squares it internally) |
| **absolute_error** | $\|y - \hat{y}\|$ | $[0, \infty)$ | `MAEloss` |

## See Also
- [MSEloss](mse-loss.md) — Uses `residual_error`.
- [MAEloss](mae-loss.md) — Uses `absolute_error`.
