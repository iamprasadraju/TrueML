# How-to: Implement Minibatch Gradient Descent

By default, the tutorials show Full-Batch Gradient Descent: the model computes the loss and gradients across the entire dataset $X$ before making a single parameter update. If your dataset contains 1,000,000 rows, a single update step will be very slow and memory-intensive.

This guide shows how to adapt the TrueML training loop to use **Minibatch Gradient Descent**, where updates occur using small chunks of data.

## When to use this guide
- Your dataset is too large to fit in memory or processes too slowly.
- You want to introduce stochastic noise to escape saddle points or local minima.
- You want faster initial convergence.

---

## 1. Creating a Batch Iterator

Because TrueML does not have built-in data loaders (like PyTorch's `DataLoader`), you must write a simple generator to yield batches of data.

```python
import numpy as np

def iterate_minibatches(X, y, batch_size, shuffle=True):
    """Yields consecutive minibatches from X and y."""
    assert X.shape[0] == y.shape[0]
    indices = np.arange(X.shape[0])
    
    if shuffle:
        np.random.shuffle(indices)
        
    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X.shape[0])
        excerpt = indices[start_idx:end_idx]
        yield X[excerpt], y[excerpt]
```

---

## 2. The Minibatch Training Loop

We now add an inner loop over the batches. An **epoch** is defined as one full pass through the dataset, meaning multiple updates will happen per epoch.

```python
from trueml.linearmodel import LinearRegression
from trueml.losses import MSEloss

# Assume X and y are already defined (e.g., 10,000 rows)
n_samples, n_features = X.shape
batch_size = 32

model = LinearRegression(n_features=n_features, lr=0.01)
loss_fn = MSEloss()

epochs = 20

for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    # Inner loop: iterate over minibatches
    for X_batch, y_batch in iterate_minibatches(X, y, batch_size, shuffle=True):
        
        # 1. Forward on the BATCH
        y_pred = model.forward(X_batch)
        
        # 2. Loss on the BATCH
        loss = loss_fn(y_batch, y_pred)
        epoch_loss += loss
        num_batches += 1
        
        # 3. Gradients on the BATCH
        dloss = loss_fn.grad(y_batch, y_pred)
        dw, db = model.grad(X_batch, dloss)
        
        # 4. Update on the BATCH
        model.backward(dw, db)
        
    # Calculate average loss across the epoch for reporting
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1:2d} | Avg Loss: {avg_loss:.4f}")
```

## Comparison with Full-Batch

| | Full-Batch | Minibatch |
|---|---|---|
| **Updates per Epoch** | 1 | `N / batch_size` |
| **Gradient Quality** | Exact average | Noisy estimate |
| **Memory Usage** | High (Entire dataset) | Low (Just the batch) |
| **Learning Rate Need** | Can be large | Must be smaller (due to noise) |

!!! tip "Troubleshooting Minibatch Convergence"
    If your loss starts wildly fluctuating or diverging to `NaN` when switching to minibatches, **lower your learning rate**. The gradient of a batch of 32 rows is much noisier than the gradient of 10,000 rows, and a high learning rate will amplify that noise, throwing the model off course.
