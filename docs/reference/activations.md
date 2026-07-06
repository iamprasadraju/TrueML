# Activations

**Module:** `trueml.activations`

Activations are element-wise nonlinearities for model forward passes. They are pure functions with no internal state, typically used inside a model's `forward()` method to transform the linear predictor into probabilities, thresholds, or other mapped spaces.

## Design Philosophy

Separating activations from models keeps each model file focused on its prediction formula and optimization mechanics, avoiding code duplication when adding new architectures. By keeping them stateless, they can be swapped or tested independently of any learned parameters.

---

## `sigmoid`

```python
trueml.activations.sigmoid(x: np.ndarray) -> np.ndarray
```

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
    from trueml.visualize import viz2d
    
    viz2d(sigmoid, x_range=(-10, 10))
    ```

---

## `linear`

```python
trueml.activations.linear(x: np.ndarray) -> np.ndarray
```

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

## Comparison Table

| Name | Formula | Range | Derivative | Use Case |
|------|---------|-------|------------|----------|
| **sigmoid** | $\frac{1}{1 + e^{-x}}$ | $(0, 1)$ | $\sigma(x)(1 - \sigma(x))$ | Binary classification probabilities |
| **linear** | $x$ | $(-\infty, \infty)$ | $1$ | Continuous regression |
| **relu** (planned) | $\max(0, x)$ | $[0, \infty)$ | $1 \text{ if } x > 0 \text{ else } 0$ | Deep network hidden layers |
| **tanh** (planned) | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | $1 - \tanh^2(x)$ | Zero-centered bounded outputs |
| **softmax** (planned) | $\frac{e^{x_i}}{\sum e^{x_j}}$ | $(0, 1)$ | $p_i(\delta_{ij} - p_j)$ | Multi-class classification |

## See Also
- [LogisticRegression](logistic-regression.md) — Uses `sigmoid` internally.
- [LinearRegression](linear-regression.md) — Effectively uses `linear` activation.
