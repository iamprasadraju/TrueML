# sigmoid

**Module:** `trueml.activations.sigmoid`

The logistic sigmoid function. Maps any real-valued number into the range $(0, 1)$, making it useful for generating probabilities in binary classification.

**Formula:**
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Derivative:**
$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

**Example:**
```python
import numpy as np
from trueml.activations import sigmoid

x = np.array([-10.0, 0.0, 10.0])
print(sigmoid(x))
# [4.53978687e-05 5.00000000e-01 9.99954602e-01]
```

!!! tip "Visualization"
    You can visualize this activation using the TrueML 2D plotting tools:
    ```python
    from trueml.activations import sigmoid
    from trueml.plots import function2d
    
    function2d(sigmoid, x_range=(-10, 10))
    ```

---

## API Reference

::: trueml.activations.sigmoid
    options:
      show_source: true
      heading_level: 3
