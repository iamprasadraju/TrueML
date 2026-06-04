# SquaredError

**Module:** `tinymlx.lossfunc.SquaredError`

The element-wise squared error loss: $L_i = (y_i - \hat{y}_i)^2$. Also known as L2 loss. Penalizes large errors quadratically.

---

## Mathematical Contract

| Phase | Method | Signature | Formula |
|-------|--------|-----------|---------|
| Forward | `__call__` | `(y, y_pred) → error` | $e_i = (y_i - \hat{y}_i)^2$ |
| Gradient | `grad` | `(X, error) → (dw, db)` | $dw = \frac{1}{n} X^\mathsf{T} \frac{\partial L}{\partial \hat{y}}$ <br> $db = \frac{1}{n} \sum_i \frac{\partial L_i}{\partial \hat{y}_i}$ |

---

## Method: `__call__`

```python
loss_fn(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Computes the element-wise squared error.

**Input:** $y \in \mathbb{R}^{n}$ — ground-truth target values.

**Input:** $\hat{y} \in \mathbb{R}^{n}$ — predicted values.

**Output:** $e \in \mathbb{R}^{n}$ — squared residuals $e_i = (y_i - \hat{y}_i)^2$.

---

## Method: `grad`

```python
loss_fn.grad(X: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, float]
```

Computes the partial derivatives of the loss with respect to the model parameters $w$ and $b$.

**Input:** $X \in \mathbb{R}^{n \times d}$ — design matrix.

**Input:** $e \in \mathbb{R}^{n}$ — the **signed** residual vector $e_i = y_i - \hat{y}_i$ (not the output of `__call__`).

**Output:** Tuple $(dw, db)$.

### Derivative Derivation

Let $e_i = y_i - \hat{y}_i$. The loss is $L_i = e_i^2$. Differentiating:

$$
\frac{\partial L_i}{\partial \hat{y}_i}
= 2 e_i \cdot \frac{\partial e_i}{\partial \hat{y}_i}
= 2 (y_i - \hat{y}_i) \cdot (-1)
= -2 (y_i - \hat{y}_i)
$$

Define $g_i = \partial L_i / \partial \hat{y}_i = -2 (y_i - \hat{y}_i)$. Then by the chain rule:

$$
\begin{aligned}
\frac{\partial L}{\partial w}
&= \frac{1}{n} X^\mathsf{T} g
= -\frac{2}{n} X^\mathsf{T} (y - \hat{y})
\\[6pt]
\frac{\partial L}{\partial b}
&= \frac{1}{n} \sum_{i=1}^{n} g_i
= -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)
\end{aligned}
$$

### Implementation

```python
def grad(self, X, error):
    n = X.shape[0]
    grad_pred = -2 * error               # ∂L/∂ŷ  (shape: n)
    dw = (X.T @ grad_pred) / n          # ∂L/∂w  (shape: d)
    db = np.mean(grad_pred)             # ∂L/∂b  (scalar)
    return dw, db
```

---

## Usage Example

```python
import numpy as np
from tinymlx.lossfunc import SquaredError

y_true = np.array([2.0, -1.0, 0.5])
y_pred = np.array([1.8, -0.7, 0.6])
X = np.random.randn(3, 4)

loss = SquaredError()
sq_error = loss(y_true, y_pred)               # [0.04, 0.09, 0.01]
signed_error = y_true - y_pred                # [0.2, -0.3, -0.1]
dw, db = loss.grad(X, signed_error)           # gradients w.r.t. w and b
```

---

## Properties

- **Magnitude-sensitive gradient**: The gradient magnitude grows linearly with the error. This makes SquaredError **sensitive to outliers** — a single large error dominates the gradient.
- **Smooth everywhere**: Unlike AbsoluteError, the squared error is infinitely differentiable, making it compatible with second-order optimization methods.
