# History

**Module:** `trueml.history`

A lightweight key-value store that records arbitrary named metrics at each training epoch. Created automatically by linear models when `history=True`.

---

## Usage Example

```python
from trueml.linear_model import LinearRegression
from trueml.losses import MSEloss
import numpy as np

model = LinearRegression(n_features=3, lr=0.01)
loss_fn = MSEloss()

X = np.random.randn(100, 3)
y = X @ np.array([1.5, -2.0, 0.5]) + 0.1

for epoch in range(100):
    y_pred = model.forward(X)
    loss = loss_fn(y, y_pred)
    dw, db = model.grad(X, loss_fn.grad(y, y_pred))
    model.backward(dw, db)
    model.history.append(epoch, loss=loss)

# Retrieve recorded metrics
print(model.history["loss"][:5])  # First 5 loss values
```

You can also visualize training history:

```python
from trueml.plots import history as plot_history
plot_history(model.history)
```

---

## API Reference

::: trueml.history.History
    options:
      show_source: true
      heading_level: 3
      members_order: source

## See Also
- [LinearRegression](linear_model/linear_regression.md) — Creates a History object automatically.
- [plot_history](plots/plot_history.md) — Plot training metrics with `plots.history()`.
