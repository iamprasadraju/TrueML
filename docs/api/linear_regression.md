# LinearRegression

**Module:** `tinymlx.linearmodel.LinearRegression`

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
| `weights` | `(n_features,)` | $\mathcal{N}(0, 0.01)$ |
| `bias` | `scalar` | `0.0` |

---

## Method: `forward`

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

**State modified:** None. `forward` is a pure function with respect to parameters.

---

## Method: `backward`

```python
model.backward(dw: np.ndarray, db: float) -> None
```

Updates the model parameters via gradient descent.

**Input:** $dw \in \mathbb{R}^{d}$ — gradient of the loss with respect to the weight vector $w$.

**Input:** $db \in \mathbb{R}$ — gradient of the loss with respect to the bias $b$.

**Update rule:**
$$
\begin{aligned}
w &\gets w - \eta \cdot dw \\
b &\gets b - \eta \cdot db
\end{aligned}
$$

**State modified:** `self.weights`, `self.bias`.

---

## Usage Example

```python
import numpy as np
from tinymlx.linearmodel import LinearRegression
from tinymlx.lossfunc import AbsoluteError

n, d = 100, 3
X = np.random.randn(n, d)
y = X @ np.array([1.5, -2.0, 0.5]) + 0.1

model = LinearRegression(n_features=d, lr=0.01)
loss = AbsoluteError()

for epoch in range(500):
    y_pred = model.forward(X)
    error = loss(y, y_pred)
    dw, db = loss.grad(X, error)
    model.backward(dw, db)

    if epoch % 50 == 0:
        mae = np.mean(np.abs(error))
        print(f"epoch {epoch:3d}  MAE = {mae:.6f}")
```

See the [Manual Gradient Descent](../recipes/manual_gradient_descent.md) recipe for a fully annotated walkthrough.
