<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="assets/trueml-dark.svg">
  <img alt="trueml logo" src="assets/trueml-light.svg" width="60%" height="50%">
</picture>

<br>
<br>


**Machine learning without hidden abstractions.**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/iamprasadraju/TrueML/actions/workflows/ci.yml/badge.svg)](https://github.com/iamprasadraju/TrueML/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

> [!CAUTION]
> **Library Under Development**
> TrueML is currently in early development. APIs are subject to change.

---

## What is TrueML?

TrueML is a Python machine learning library built on a single, uncompromising principle: **every mathematical operation in the learning pipeline is a first-class function you invoke explicitly.** 

There is no `.fit()`. There is no hidden state. There are no implicit optimizers or black-box solvers. 

Instead, TrueML provides the primitive mathematical operations (forward passes, loss functions, Jacobians, and parameter updates) and **you write the training loop**. 

If you are a student learning how gradients flow, a researcher auditing an optimization landscape, or a practitioner who wants a maximally transparent baseline before layering complexity, TrueML is built for you.

---

## Quick Start

The core of TrueML is the four-step explicit pipeline: `forward` $\rightarrow$ `loss` $\rightarrow$ `gradient` $\rightarrow$ `backward`.

```python
import numpy as np
from trueml.linear_model import LinearRegression
from trueml.losses import MSELoss

# 1. Prepare data
X = np.random.randn(100, 3)
y = X @ np.array([1.5, -2.0, 0.5]) + 0.1

# 2. Initialize Model and Loss
model = LinearRegression(lr=0.01)
loss_fn = MSELoss()

# 3. The Explicit Training Loop
for epoch in range(500):
    
    # Step 1: Forward Pass (ŷ = Xw + b)
    y_pred = model.forward(X)
    
    # Step 2: Loss Computation (L = mean((y - ŷ)²))
    loss = loss_fn(y, y_pred)
    
    # Step 3: Gradient Computation (∂L/∂ŷ)
    dloss = loss_fn.grad(y, y_pred)
    
    # Step 4: Backward Update (w ← w - η · ∂L/∂w)
    model.backward(dloss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")
```

---

## Features

- **Transparent Calculus:** The multivariable chain rule is exposed directly in code. You pass the derivative of the loss (`dloss`) into the model's Jacobian (`model.grad()`) explicitly.
- **Strictly Stateless:** Models hold `weights` and `bias`, but **never** cache data (`X_train_` or `y_train_`). You must supply the data every time you compute a forward pass or gradient.
- **Auditable Math:** Every intermediate step (`y_pred`, `loss`, `dloss`, `dw`) is a standard NumPy array. You can intercept, print, clip, or plot them at any time.

---

## Installation

TrueML requires **Python 3.11+**. Core dependencies are `numpy` and `networkx` (for computation graph visualization).

It is available on PyPI (but still in the developing stage):

```bash
pip install trueml
```

Alternatively, you can install it from source:

```bash
git clone https://github.com/iamprasadraju/trueml.git
cd trueml
pip install -e .
```

---

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/iamprasadraju/trueml.git
cd trueml
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Build docs
uv sync --group docs
uv run mkdocs serve
```

---

## Contributing

Contributions are welcome! If you want to add new models, loss functions, or mathematical tools, please ensure they strictly adhere to the "No-Abstraction Philosophy":
1. No hidden state.
2. No implicit optimizations.
3. Every operation must clearly map to a mathematical equation.
4. **Don't push AI generated code.** We value human-written, deeply understood mathematical implementations.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
