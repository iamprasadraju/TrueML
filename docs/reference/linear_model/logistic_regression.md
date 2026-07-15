# LogisticRegression

**Module:** `trueml.linear_model.logistic_regression`

A binary logistic classifier of the form $p = \sigma(Xw + b)$, where $\sigma$ is the sigmoid function. Supports explicit gradient-descent training via the `backward` method.

## Mathematical Contract

| Phase | Method | Signature | Formula |
|-------|--------|-----------|---------| 
| Forward | `forward` | `(X) → prob` | $p = \sigma(Xw + b)$ |
| Update | `backward` | `(dw, db) → None` | $w \gets w - \eta \cdot dw$ <br> $b \gets b - \eta \cdot db$ |

## Derivative Derivation

By the chain rule, the gradient of the loss $L$ with respect to the weights $w$ is:
$$
dw = X^\mathsf{T} \frac{\partial L}{\partial z}
$$
And for the bias:
$$
db = \sum_{i=1}^{n} \frac{\partial L_i}{\partial z_i}
$$

## Binary Cross-Entropy Loss Pairing

`LogisticRegression` is typically paired with Binary Cross-Entropy (BCE) loss for binary classification ($y \in \{0, 1\}$). Since TrueML exposes primitive operations, we compute the BCE gradient directly.

The BCE loss is defined as:
$$
L = -\frac{1}{n} \sum_{i=1}^{n} \bigl[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \bigr]
$$

The gradient of this loss with respect to the logit $z$ simplifies elegantly to:
$$
\frac{\partial L}{\partial z} = \frac{1}{n} (p - y)
$$

We pass this $\frac{\partial L}{\partial z}$ as the `loss_gradient` into `model.grad()`.

## Usage Example

```python
import numpy as np
from trueml.linear_model import LogisticRegression

# 1. Generate synthetic binary classification data
n, d = 100, 2
X = np.random.randn(n, d)
# Decision boundary at 1.5*x1 - 2.0*x2 + 0.5 = 0
logits = X @ np.array([1.5, -2.0]) + 0.5
probs = 1 / (1 + np.exp(-logits))
y = (probs >= 0.5).astype(float)

# 2. Initialize model
model = LogisticRegression(n_features=d, lr=0.1)

# 3. Training loop with implicit BCE loss
for epoch in range(1, 501):
    p = model.forward(X)
    
    # BCE loss gradient w.r.t logits (z)
    loss_grad = (p - y) / n
    
    # Compute gradients and update
    dw, db = model.grad(X, loss_grad)
    model.backward(dw, db)

    if epoch % 100 == 0:
        bce_loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
        acc = np.mean((p >= 0.5) == y)
        print(f"epoch {epoch:3d}  BCE = {bce_loss:.4f}  Acc = {acc:.2f}")
```

---

## API Reference

::: trueml.linear_model.logistic_regression
    options:
      show_source: true
      heading_level: 3
      members_order: source
