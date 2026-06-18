# How to Implement Minibatch Gradient Descent

This guide shows you how to modify a full-batch training loop to process data in minibatches, which is essential for scaling to larger datasets.

## When to use this guide

Use this when your dataset is too large to fit in memory for a single gradient computation, or when you want the faster convergence that noisy minibatch updates provide.

## Before you start

- Familiarity with the full-batch training loop (see [Manual Gradient Descent](./manual-gradient-descent.md))
- A dataset loaded as numpy arrays `X` and `y`

## Context

In full-batch gradient descent, the gradient is computed over all observations before each update. For large $n$, this is expensive and the gradient is deterministic — it always follows the steepest direction for the full dataset.

Minibatch gradient descent estimates the gradient from a random subset of observations. The estimate is noisy, but each update is much cheaper, and the noise can help escape shallow local minima.

## Steps

### 1. Define the batch iterator

```python
import numpy as np
from trueml.linearmodel import LinearRegression
from trueml.lossfunc import AbsoluteError

def iterate_minibatches(X, y, batch_size, shuffle=True):
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
```

### 2. Initialize model and loss

```python
model = LinearRegression(n_features=X.shape[1], lr=0.01)
loss_fn = AbsoluteError()
```

### 3. Run the minibatch training loop

```python
epochs = 50
batch_size = 32

for epoch in range(1, epochs + 1):
    epoch_losses = []
    for X_batch, y_batch in iterate_minibatches(X, y, batch_size):
        y_pred = model.forward(X_batch)
        error = loss_fn(y_batch, y_pred)
        dw, db = loss_fn.grad(X_batch, error)
        model.backward(dw, db)
        epoch_losses.append(np.mean(np.abs(error)))

    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  mean MAE = {np.mean(epoch_losses):.6f}")
```

### 4. Full-batch comparison

To see the effect of minibatch noise, compare with the full-batch equivalent:

```python
model_fb = LinearRegression(n_features=X.shape[1], lr=0.01)
epochs_fb = 50

for epoch in range(1, epochs_fb + 1):
    y_pred = model_fb.forward(X)
    error = loss_fn(y, y_pred)
    dw, db = loss_fn.grad(X, error)
    model_fb.backward(dw, db)

    if epoch % 10 == 0:
        mae = np.mean(np.abs(error))
        print(f"epoch {epoch:3d}  full-batch MAE = {mae:.6f}")
```

## Troubleshooting

**Problem: Minibatch training is noisier than full-batch**
Solution: This is expected. The gradient from a small batch is a noisy estimate of the true gradient. Increasing batch size reduces noise but increases computation per update.

**Problem: Model does not converge with small batch sizes**
Solution: Try increasing the learning rate. Noisy gradients need slightly larger steps to make progress. Alternatively, increase batch size to reduce gradient variance.

## Variations

**SGD (batch size = 1):**
Set `batch_size = 1`. Each update uses a single observation. This is maximally noisy but maximally fast per update.

**Shuffling every epoch:**
The `iterate_minibatches` function above shuffles every pass through the data. To disable shuffling (for debugging), pass `shuffle=False`.

**Tracking the full-batch loss:**
To get a clean evaluation metric each epoch, compute the full-batch MAE after all minibatch updates:

```python
for epoch in range(1, epochs + 1):
    for X_batch, y_batch in iterate_minibatches(X, y, batch_size):
        y_pred = model.forward(X_batch)
        error = loss_fn(y_batch, y_pred)
        dw, db = loss_fn.grad(X_batch, error)
        model.backward(dw, db)

    y_pred_full = model.forward(X)
    full_mae = np.mean(np.abs(loss_fn(y, y_pred_full)))
    print(f"epoch {epoch:3d}  full MAE = {full_mae:.6f}")
```

## Related guides

- [How to Train on a Real Dataset](./train-on-real-data.md) — applying minibatch GD to Housing.csv
- [Manual Gradient Descent](./manual-gradient-descent.md) — full-batch baseline
- [About Gradient Descent](../explanation/about-gradient-descent.md) — theory behind minibatch convergence
