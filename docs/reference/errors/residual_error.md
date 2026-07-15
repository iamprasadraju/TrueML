# residual_error

**Module:** `trueml.errors.residual_error`

Computes the raw, signed residual error between actual and predicted values.

**Formula:**
$$
e = y_{true} - y_{pred}
$$

**Properties:**
- Positive error indicates underprediction ($y_{true} > y_{pred}$).
- Negative error indicates overprediction ($y_{true} < y_{pred}$).
- Errors can cancel out if aggregated naively.

**Example:**
```python
from trueml.errors import residual_error
import numpy as np

y_true = np.array([10, 20])
y_pred = np.array([8, 25])

print(residual_error(y_true, y_pred))
# [2, -5]
```

---

## API Reference

::: trueml.errors.residual_error
    options:
      show_source: true
      heading_level: 3
