# SignedError

**Module:** `trueml.lossfunc.SignedError`

The element-wise signed error: $e_i = y_i - \hat{y}_i$. Measures raw deviation while preserving the sign of the residual. Callable via the class instance.

---

## Mathematical Contract

| Phase | Method | Signature | Formula |
|-------|--------|-----------|---------|
| Forward | `__call__` | `(y, y_pred) → error` | $e_i = y_i - \hat{y}_i$ |

---

## Method: `__call__`

```python
loss_fn(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Computes the element-wise signed error.

**Input:** $y \in \mathbb{R}^{n}$ — ground-truth target values.

**Input:** $\hat{y} \in \mathbb{R}^{n}$ — predicted values.

**Output:** $e \in \mathbb{R}^{n}$ — signed residuals $e_i = y_i - \hat{y}_i$.

**Properties:**

- Positive error means under-prediction ($y_i > \hat{y}_i$).
- Negative error means over-prediction ($y_i < \hat{y}_i$).
- Errors can cancel across observations (mean signed error may be zero even when predictions are poor).

---

## Usage Example

```python
import numpy as np
from trueml.lossfunc import SignedError

y_true = np.array([2.0, -1.0, 0.5])
y_pred = np.array([1.8, -0.7, 0.6])

loss = SignedError()
error = loss(y_true, y_pred)    # [0.2, -0.3, -0.1]
```

---

## Notes

- `SignedError` has no `grad` method. It is used for inspecting residual signs, not for gradient-based training.
- For gradient computation, use `AbsoluteError` or `SquaredError`, both of which provide a `grad(X, error)` method.
- The signed error is useful for detecting systematic bias in predictions: if all errors have the same sign, the model is consistently over- or under-predicting.

## Related references

- [AbsoluteError](./absolute-error.md) — L1 loss with gradient
- [SquaredError](./squared-error.md) — L2 loss with gradient
