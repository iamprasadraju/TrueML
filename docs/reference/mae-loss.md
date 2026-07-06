# MAEloss

**Module:** `trueml.losses.MAEloss`

The Mean Absolute Error (MAE) loss, also known as L1 loss. It measures the average magnitude of the prediction errors without considering their direction.

---

## Mathematical Contract

| Phase | Method | Signature | Formula |
|-------|--------|-----------|---------|
| Loss | `__call__` | `(y_true, y_pred) → scalar` | $L = \frac{1}{n}\sum \|y_i - \hat{y}_i\|$ |
| Gradient | `grad` | `(y_true, y_pred) → array` | $\frac{\partial L}{\partial \hat{y}} = \text{sign}(\hat{y} - y)$ |
| Surface | `surface` | `(y_true, y_pred) → array` | $e_i = \|y_i - \hat{y}_i\|$ |

---

## Methods

### `__call__`

```python
loss_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float
```

Computes the scalar mean absolute error.

**Formula:**
$$
L = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

---

### `grad`

```python
loss_fn.grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Computes the gradient (or subgradient) of the MAE loss with respect to the predictions ($\hat{y}$).

**Formula:**
$$
\frac{\partial L}{\partial \hat{y}_i} = \text{sign}(\hat{y}_i - y_i)
$$

**Derivative Derivation:**
Let the overall loss be $L = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$. Since this function is linear with respect to the prediction (except at the exact origin), the derivative ignores the $1/n$ scaling if we treat the gradient as a sum update, but in TrueML, the gradient is defined via the subgradient of the absolute value function:
$$
\frac{\partial}{\partial \hat{y}_i} |y_i - \hat{y}_i| = \text{sign}(\hat{y}_i - y_i)
$$
Where $\text{sign}(x)$ is $1$ if $x > 0$, $-1$ if $x < 0$, and $0$ if $x = 0$.

!!! tip "Subgradient at zero"
    The absolute value function is not differentiable at $x=0$. In practice, `np.sign(0)` returns `0`, which naturally acts as the subgradient at this point, effectively halting the update for perfectly predicted samples.

---

### `surface`

```python
loss_fn.surface(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Returns the element-wise absolute error $|y_i - \hat{y}_i|$. Useful for 3D visualizations of the loss landscape.

---

## Properties

- **Constant Gradient Magnitude:** Because the gradient relies on the `sign` function, its magnitude is always $\pm 1$ (or 0). The model takes the exact same size step whether the prediction is off by 1 unit or 1000 units.
- **Robustness to Outliers:** MAEloss is far more robust to outliers than [MSEloss](mse-loss.md). A single anomalous data point will only contribute a constant $\pm 1$ to the gradient, preventing the outlier from dominating the training update.

---

## Usage Example

```python
import numpy as np
from trueml.linearmodel import LinearRegression
from trueml.losses import MAEloss

X = np.random.randn(50, 2)
y_true = X @ np.array([2.5, -1.0]) + 0.5

model = LinearRegression(n_features=2, lr=0.1)
loss_fn = MAEloss()

for epoch in range(100):
    y_pred = model.forward(X)
    
    # 1. Compute loss value
    loss = loss_fn(y_true, y_pred)
    
    # 2. Compute loss gradient
    dloss = loss_fn.grad(y_true, y_pred)
    
    # 3. Chain rule
    dw, db = model.grad(X, dloss)
    
    # 4. Update
    model.backward(dw, db)
```

## See Also
- [MSEloss](mse-loss.md) — Mean Squared Error loss.
- [errors](errors.md) — Underlying error metrics used by these losses.
