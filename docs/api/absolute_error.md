# AbsoluteError

**Module:** `tinymlx.lossfunc.AbsoluteError`

The element-wise absolute error loss: $L_i = |y_i - \hat{y}_i|$. Also known as L1 loss. Callable via the class instance.

---

## Mathematical Contract

| Phase | Method | Signature | Formula |
|-------|--------|-----------|---------|
| Forward | `__call__` | `(y, y_pred) → error` | $e_i = \|y_i - \hat{y}_i\|$ |
| Gradient | `grad` | `(X, error) → (dw, db)` | $dw = \frac{1}{n} X^\mathsf{T} \frac{\partial L}{\partial \hat{y}}$ <br> $db = \frac{1}{n} \sum_i \frac{\partial L_i}{\partial \hat{y}_i}$ |

---

## Method: `__call__`

```python
loss_fn(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Computes the element-wise absolute error.

**Input:** $y \in \mathbb{R}^{n}$ — ground-truth target values.

**Input:** $\hat{y} \in \mathbb{R}^{n}$ — predicted values (e.g. output of `LinearRegression.forward`).

**Output:** $e \in \mathbb{R}^{n}$ — absolute residuals $e_i = |y_i - \hat{y}_i|$.

---

## Method: `grad`

```python
loss_fn.grad(X: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, float]
```

Computes the partial derivatives of the loss with respect to the model parameters $w$ and $b$, via the chain rule through the model's prediction $\hat{y}$.

**Input:** $X \in \mathbb{R}^{n \times d}$ — design matrix (same $X$ passed to `forward`).

**Input:** $e \in \mathbb{R}^{n}$ — the error vector produced by `__call__`.

**Output:** Tuple $(dw, db)$:
- $dw \in \mathbb{R}^{d}$ — gradient with respect to weights.
- $db \in \mathbb{R}$ — gradient with respect to bias.

### Derivative Derivation

Let $L_i = |e_i|$ where $e_i = y_i - \hat{y}_i$. The loss is differentiable almost everywhere, with subgradient at $e_i = 0$:

$$
\frac{\partial L_i}{\partial \hat{y}_i}
= \frac{\partial}{\partial \hat{y}_i} |y_i - \hat{y}_i|
= -\text{sign}(y_i - \hat{y}_i)
= -\text{sign}(e_i)
$$

where

$$
\text{sign}(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x = 0 \\
-1 & \text{if } x < 0
\end{cases}
$$

Define the **gradient prediction direction** vector $g \in \mathbb{R}^{n}$:

$$
g_i = \frac{\partial L_i}{\partial \hat{y}_i}
$$

By the chain rule applied to $\hat{y} = Xw + b$:

$$
\begin{aligned}
\frac{\partial L}{\partial w}
&= \frac{1}{n} \sum_{i=1}^{n} \frac{\partial L_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial w}
= \frac{1}{n} \sum_{i=1}^{n} g_i \cdot X_i^\mathsf{T}
= \frac{1}{n} X^\mathsf{T} g
\\[6pt]
\frac{\partial L}{\partial b}
&= \frac{1}{n} \sum_{i=1}^{n} \frac{\partial L_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial b}
= \frac{1}{n} \sum_{i=1}^{n} g_i
= \text{mean}(g)
\end{aligned}
$$

### Implementation

```python
def grad(self, X, error):
    grad_pred = -np.sign(error)         # ∂L/∂ŷ  (shape: n)
    n = X.shape[0]
    dw = (X.T @ grad_pred) / n          # ∂L/∂w  (shape: d)
    db = np.mean(grad_pred)             # ∂L/∂b  (scalar)
    return dw, db
```

---

## Usage Example

```python
import numpy as np
from tinymlx.lossfunc import AbsoluteError

y_true = np.array([2.0, -1.0, 0.5])
y_pred = np.array([1.8, -0.7, 0.6])
X = np.random.randn(3, 4)

loss = AbsoluteError()
error = loss(y_true, y_pred)     # [0.2, 0.3, 0.1]
dw, db = loss.grad(X, error)     # gradients w.r.t. w and b
```

---

## Properties

- **Scale-invariant gradient**: The magnitude of $\partial L/\partial \hat{y}$ is constant ($\pm 1$) regardless of the error size. This makes AbsoluteError **robust to outliers** but can produce vanishing or oscillating gradients when errors are small.
- **Subgradient at zero**: `np.sign(0)` returns `0` in NumPy, which corresponds to the subgradient convention.
