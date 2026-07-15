# linear

**Module:** `trueml.activations.linear`

The linear (identity) activation function. Returns the input unchanged.

**Formula:**
$$
f(x) = x
$$

**Derivative:**
$$
f'(x) = 1
$$

**Example:**
```python
from trueml.activations import linear
import numpy as np

x = np.array([-5.0, 3.2])
print(linear(x))
# [-5.0, 3.2]
```

---

## API Reference

::: trueml.activations.linear
    options:
      show_source: true
      heading_level: 3
