# LinearRegression

**Module:** `trueml.linearmodel.LinearRegression`

A linear predictor of the form $\hat{y} = Xw + b$. Supports explicit gradient-descent training via the `backward` method.

---

## Mathematical Contract

| Phase | Method | Signature | Formula |
|-------|--------|-----------|---------|
| Forward | `forward` | `(X) → y_pred` | $\hat{y} = Xw + b$ |
| Update | `backward` | `(dw, db) → None` | $w \gets w - \eta \cdot dw$ <br> $b \gets b - \eta \cdot db$ |

---

## Constructor

```python
LinearRegression(n_features: int, lr: float = 0.01)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_features` | `int` | — | Dimensionality $d$ of the input space. Must equal the number of columns in $X$. |
| `lr` | `float` | `0.01` | Learning rate $\eta$ for the gradient descent update. |

### Initial State

| Variable | Shape | Initial Value |
|----------|-------|---------------|
| `weights` | `(n_features,)` | $\mathcal{N}(0, 0.01^2)$ |
| `bias` | `float` | `0.0` |

---

## Methods

### `forward`

```python
model.forward(X: np.ndarray) -> np.ndarray
```

Computes the linear predictor.

**Input:** $X \in \mathbb{R}^{n \times d}$ — design matrix of $n$ observations each with $d$ features.

**Output:** $\hat{y} \in \mathbb{R}^{n}$ — predicted values.

**Formula:**
$$
\hat{y}_i = \sum_{j=1}^{d} X_{ij} w_j + b \quad \text{for } i = 1, \ldots, n
$$

Or, in vectorized form:
$$
\hat{y} = X w + b
$$

!!! note "State Modified"
    None. `forward` is a pure function with respect to parameters.

---

### `grad`

```python
model.grad(X: np.ndarray, loss_gradient: np.ndarray) -> tuple[np.ndarray, float]
```

Computes the gradients of model parameters using the upstream loss gradient.

**Input:** $X \in \mathbb{R}^{n \times d}$ — design matrix used during the forward pass.

**Input:** $\frac{\partial L}{\partial \hat{y}} \in \mathbb{R}^{n}$ (`loss_gradient`) — gradient of the loss with respect to predictions.

**Output:** Tuple $(dw, db)$ where:
- $dw \in \mathbb{R}^{d}$ — gradient with respect to the weights.
- $db \in \mathbb{R}$ — gradient with respect to the bias.

**Derivative Derivation:**

By the chain rule, the gradient of the loss $L$ with respect to the weights $w$ is:
$$
\frac{\partial L}{\partial w} = \frac{\partial \hat{y}}{\partial w} \frac{\partial L}{\partial \hat{y}}
$$
Since $\hat{y} = Xw + b$, the Jacobian $\frac{\partial \hat{y}}{\partial w} = X$. Therefore:
$$
dw = X^\mathsf{T} \frac{\partial L}{\partial \hat{y}}
$$
Similarly, for the bias:
$$
db = \sum_{i=1}^{n} \frac{\partial L_i}{\partial \hat{y}_i}
$$

!!! note "State Modified"
    None. `grad` only computes the derivatives.

---

### `backward`

```python
model.backward(dw: np.ndarray, db: float) -> None
```

Updates the model parameters via gradient descent optimization.

**Input:** $dw \in \mathbb{R}^{d}$ — gradient of the loss with respect to the weights.

**Input:** $db \in \mathbb{R}$ — gradient of the loss with respect to the bias.

**Update rule:**
$$
\begin{aligned}
w &\gets w - \eta \cdot dw \\
b &\gets b - \eta \cdot db
\end{aligned}
$$

!!! note "State Modified"
    `self.weights`, `self.bias`.

---

## Usage Example

```python
import numpy as np
from trueml.linearmodel import LinearRegression
from trueml.losses import MSEloss

# 1. Generate synthetic data
n, d = 100, 3
X = np.random.randn(n, d)
y = X @ np.array([1.5, -2.0, 0.5]) + 0.1

# 2. Initialize model and loss
model = LinearRegression(n_features=d, lr=0.01)
loss_fn = MSEloss()

# 3. Training loop
for epoch in range(1, 501):
    y_pred = model.forward(X)
    loss_value = loss_fn(y, y_pred)
    
    # Gradients
    loss_grad = loss_fn.grad(y, y_pred)
    dw, db = model.grad(X, loss_grad)
    
    # Update
    model.backward(dw, db)

    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  MSE = {loss_value:.6f}")
```

---

## Notes

- The `grad` method expects `loss_gradient` (e.g., from `MSEloss.grad()`). It does not compute the error itself.
- Unlike scikit-learn's `LinearRegression.fit()`, there is no closed-form solution (Normal Equation) computed here. This model strictly uses iterative gradient descent.

## See Also
- [LogisticRegression](logistic-regression.md) — For binary classification tasks.
- [MSEloss](mse-loss.md) — Mean Squared Error loss.
- [MAEloss](mae-loss.md) — Mean Absolute Error loss.
