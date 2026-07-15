# absolute_error

**Module:** `trueml.errors.absolute_error`

Computes the absolute magnitude of the error, discarding the sign.

**Formula:**
$$
e = |y_{true} - y_{pred}|
$$

**Properties:**
- Strictly non-negative.
- Accurately reflects the magnitude of the discrepancy without allowing positive and negative errors to cancel out.

**Example:**
```python
from trueml.errors import absolute_error
import numpy as np

y_true = np.array([10, 20])
y_pred = np.array([8, 25])

print(absolute_error(y_true, y_pred))
# [2, 5]
```

---

## API Reference

::: trueml.errors.absolute_error
    options:
      show_source: true
      heading_level: 3
