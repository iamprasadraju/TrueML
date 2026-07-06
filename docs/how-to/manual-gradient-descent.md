# How-to: Manual Gradient Descent

While TrueML provides a `.backward(dw, db)` method to update weights, you might occasionally want to intercept the gradients and apply the update manually. This is useful for implementing custom optimizers (like Momentum or Adam), applying weight decay (L2 regularization), or simply for educational purposes to prove that there is no magic happening inside the `LinearRegression` class.

This guide walks you through bypassing the `.backward()` method and updating the parameters yourself.

## When to use this guide
- You want to implement a custom optimizer.
- You want to apply manual weight constraints (e.g., non-negative weights).
- You are debugging an exploding gradient and want to apply gradient clipping manually.

---

## The Standard Loop

Normally, a TrueML training loop looks like this:

```python
from trueml.linearmodel import LinearRegression
from trueml.losses import MSEloss
import numpy as np

X = np.random.randn(10, 2)
y = np.random.randn(10)

model = LinearRegression(n_features=2, lr=0.01)
loss_fn = MSEloss()

# Standard Forward, Grad, Backward
y_pred = model.forward(X)
dloss = loss_fn.grad(y, y_pred)
dw, db = model.grad(X, dloss)

model.backward(dw, db)  # <--- TrueML applies the update
```

## The Manual Loop

To apply the update manually, we simply take the gradients returned by `model.grad()` and apply the gradient descent formula ourselves:
$$w \gets w - \eta \cdot dw$$

```python
# Initialize as usual
model = LinearRegression(n_features=2, lr=0.01)

# Define your learning rate locally
learning_rate = 0.01

for epoch in range(100):
    # 1 & 2: Forward and Loss Gradient (Unchanged)
    y_pred = model.forward(X)
    dloss = loss_fn.grad(y, y_pred)
    
    # 3: Extract parameter gradients
    dw, db = model.grad(X, dloss)
    
    # ----------------------------------------------------
    # 4: MANUAL UPDATE
    # Do not call model.backward()
    # Modify the model's state directly!
    # ----------------------------------------------------
    
    model.weights = model.weights - (learning_rate * dw)
    model.bias = model.bias - (learning_rate * db)
```

!!! note "Accessing State"
    Because TrueML embraces transparency, `model.weights` and `model.bias` are standard public NumPy arrays. You are explicitly encouraged to read and write them.

---

## Example: Adding Weight Decay (L2 Regularization)

By applying the update manually, we can easily add custom logic. For instance, L2 regularization penalizes large weights by decaying them slightly towards zero on every step.

$$w \gets w - \eta \cdot dw - \eta \cdot \lambda \cdot w$$

```python
lambda_l2 = 0.005  # Regularization strength
learning_rate = 0.01

for epoch in range(100):
    y_pred = model.forward(X)
    dloss = loss_fn.grad(y, y_pred)
    dw, db = model.grad(X, dloss)
    
    # Manual update with L2 Weight Decay
    model.weights = model.weights - (learning_rate * dw) - (learning_rate * lambda_l2 * model.weights)
    
    # Bias is typically not regularized
    model.bias = model.bias - (learning_rate * db)
```

## Summary

TrueML's stateless design means the model class is just a container for `weights`, `bias`, and the math required to compute forward passes and Jacobians. The optimization step is completely decoupled. If `model.backward()` doesn't do what you need, you simply don't call it.
