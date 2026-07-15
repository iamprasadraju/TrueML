# MSEloss

**Module:** `trueml.losses.mean_squared_error`

The Mean Squared Error (MSE) loss, also known as L2 loss. It measures the average of the squares of the errors, penalizing larger prediction errors more heavily than smaller ones.

## Mathematical Contract

| Phase | Method | Signature | Formula |
|-------|--------|-----------|---------| 
| Loss | `__call__` | `(y_true, y_pred) → scalar` | $L = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$ |
| Gradient | `grad` | `(y_true, y_pred) → array` | $\frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)$ |
| Surface | `surface` | `(y_true, y_pred) → array` | $e_i^2 = (y_i - \hat{y}_i)^2$ |

## Derivative Derivation

Let the overall loss be $L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$. 
Differentiating $L$ with respect to a single prediction $\hat{y}_i$:
$$
\begin{aligned}
\frac{\partial L}{\partial \hat{y}_i} &= \frac{1}{n} \cdot \frac{\partial}{\partial \hat{y}_i} (y_i - \hat{y}_i)^2 \\
&= \frac{1}{n} \cdot 2(y_i - \hat{y}_i) \cdot (-1) \\
&= \frac{2}{n}(\hat{y}_i - y_i)
\end{aligned}
$$

!!! note "The $2/n$ factor"
    Some frameworks scale MSE by $\frac{1}{2n}$ to cancel out the $2$ during differentiation. TrueML preserves the exact mathematical definition of the mean squared error, so the $2/n$ factor is explicitly returned in the gradient.

## Properties

- **Sensitivity to Outliers:** Because the errors are squared, an observation off by 10 units is penalized 100 times worse than an observation off by 1 unit. This causes the gradient magnitude to grow linearly with the error, making MSE highly sensitive to outliers.
- **Smoothness:** MSE is smooth and infinitely differentiable everywhere, allowing optimization to naturally slow down (take smaller steps) as predictions approach the targets.

## Usage Example

```python
import numpy as np
from trueml.linear_model import LinearRegression
from trueml.losses import MSEloss

X = np.random.randn(50, 2)
y_true = X @ np.array([2.5, -1.0]) + 0.5

model = LinearRegression(n_features=2, lr=0.1)
loss_fn = MSEloss()

for epoch in range(100):
    y_pred = model.forward(X)
    
    # 1. Compute loss value
    loss = loss_fn(y_true, y_pred)
    
    # 2. Compute loss gradient w.r.t predictions
    dloss = loss_fn.grad(y_true, y_pred)
    
    # 3. Chain rule: gradients w.r.t model parameters
    dw, db = model.grad(X, dloss)
    
    # 4. Update
    model.backward(dw, db)
```

---

## API Reference

::: trueml.losses.mean_squared_error
    options:
      show_source: true
      heading_level: 3
      members_order: source
