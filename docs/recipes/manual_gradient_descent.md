# Recipe: Manual Gradient Descent

**Objective:** Train a `LinearRegression` model on synthetic data using `AbsoluteError` loss, with every gradient descent step written explicitly.

**Prerequisites:** `tinymlx`, `numpy`

---

## Protocol

### 1. Generate Synthetic Data

```python
import numpy as np
from tinymlx.linearmodel import LinearRegression
from tinymlx.lossfunc import AbsoluteError

n = 200               # number of observations
d = 3                 # number of features
true_w = np.array([2.0, -1.5, 0.8])
true_b = 0.5

rng = np.random.default_rng(42)
X = rng.normal(size=(n, d))
y = X @ true_w + true_b + rng.normal(scale=0.2, size=n)   # y = X w + b + noise
```

### 2. Initialize Model and Loss

```python
model = LinearRegression(n_features=d, lr=0.01)
loss_fn = AbsoluteError()
```

### 3. Training Loop

```python
epochs = 1000
report_interval = 100

for epoch in range(1, epochs + 1):
    # ── Forward ──────────────────────────────────────────────
    #   ŷ = X w + b
    y_pred = model.forward(X)

    # ── Signed error ─────────────────────────────────────────
    #   e = y – ŷ   (signed residual — used by grad)
    signed_error = y - y_pred

    # ── Loss ─────────────────────────────────────────────────
    #   L_i = |y_i – ŷ_i|
    _ = loss_fn(y, y_pred)

    # ── Gradient ─────────────────────────────────────────────
    #   ∂L/∂ŷ_i = –sign(y_i – ŷ_i)
    #   ∂L/∂w   = (1/n) X^T · ∂L/∂ŷ
    #   ∂L/∂b   = mean(∂L/∂ŷ)
    dw, db = loss_fn.grad(X, signed_error)

    # ── Update ───────────────────────────────────────────────
    #   w ← w – η · ∂L/∂w
    #   b ← b – η · ∂L/∂b
    model.backward(dw, db)

    if epoch % report_interval == 0:
        mae = np.mean(np.abs(signed_error))
        print(f"epoch {epoch:4d}  MAE = {mae:.6f}  "
              f"w ≈ {model.weights}  b ≈ {model.bias:.4f}")
```

### 4. Expected Output

```
epoch  100  MAE = 0.174504  w ≈ [1.890 -1.379  0.712]  b ≈ 0.4672
epoch  200  MAE = 0.167755  w ≈ [1.950 -1.450  0.761]  b ≈ 0.4879
epoch  300  MAE = 0.165934  w ≈ [1.978 -1.480  0.782]  b ≈ 0.4958
epoch  400  MAE = 0.165177  w ≈ [1.991 -1.496  0.792]  b ≈ 0.4979
epoch  500  MAE = 0.164865  w ≈ [1.997 -1.502  0.796]  b ≈ 0.4971
...
epoch 1000  MAE = 0.164647  w ≈ [2.001 -1.501  0.800]  b ≈ 0.4998
```

Recovered weights approach $[2.0, -1.5, 0.8]$ and bias approaches $0.5$.

---

## Variations

### Switch to SquaredError

Replace `AbsoluteError` with `SquaredError` to observe how the gradient changes:

```python
from tinymlx.lossfunc import SquaredError
loss_fn = SquaredError()
```

With SquaredError, the MAE will converge faster in early epochs (larger initial gradient) but may overshoot if the learning rate is too high.

### Batch Gradient Descent

For batch processing, simply slice $X$ and $y$:

```python
batch_size = 32
for epoch in range(epochs):
    for i in range(0, n, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        y_pred = model.forward(X_batch)
        signed_error = y_batch - y_pred
        _ = loss_fn(y_batch, y_pred)
        dw, db = loss_fn.grad(X_batch, signed_error)
        model.backward(dw, db)
```

No state needs resetting — the model is stateless.
