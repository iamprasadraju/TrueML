# LinearRegression

**Module:** `trueml.linear_model.linear_regression`

A linear predictor of the form $\hat{y} = Xw + b$. Supports explicit gradient-descent training via the `backward` method.

## Mathematical Contract

| Phase | Method | Signature | Formula |
|-------|--------|-----------|---------| 
| Forward | `forward` | `(X) → y_pred` | $\hat{y} = Xw + b$ |
| Update | `backward` | `(dw, db) → None` | $w \gets w - \eta \cdot dw$ <br> $b \gets b - \eta \cdot db$ |

## Derivative Derivation

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

## Usage Example

```python
import numpy as np
from trueml.linear_model import LinearRegression
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

## Notes
- The `grad` method expects `loss_gradient` (e.g., from `MSEloss.grad()`). It does not compute the error itself.
- Unlike scikit-learn's `LinearRegression.fit()`, there is no closed-form solution (Normal Equation) computed here. This model strictly uses iterative gradient descent.

---

## API Reference

::: trueml.linear_model.linear_regression
    options:
      show_source: true
      heading_level: 3
      members_order: source
