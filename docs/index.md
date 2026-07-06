# TrueML

**Machine learning without hidden abstractions.**

*Version 0.0.1* · [GitHub](https://github.com/iamprasadraju/trueml) · [API Reference](reference/linear-regression.md)

---

## What is TrueML?

TrueML is a Python library that exposes every mathematical operation in the machine learning pipeline as a first-class function you invoke explicitly. There is no `.fit()`. There is no hidden state. The user writes the training loop, keeping every mathematical operation visible and auditable.

```python
import numpy as np
from trueml.linearmodel import LinearRegression
from trueml.losses import MSEloss

model = LinearRegression(n_features=3, lr=0.01)
loss_fn = MSEloss()

for epoch in range(500):
    y_pred = model.forward(X)               # ŷ = Xw + b
    loss = loss_fn(y, y_pred)               # L = mean((y - ŷ)²)
    dloss = loss_fn.grad(y, y_pred)         # ∂L/∂ŷ = (2/n)(ŷ - y)
    dw, db = model.grad(X, dloss)           # ∂L/∂w = Xᵀ · ∂L/∂ŷ
    model.backward(dw, db)                  # w ← w - η · ∂L/∂w
```

Every step maps directly to a mathematical equation. Every intermediate value is a NumPy array you can print, plot, or modify.

---

## The Four-Step Pipeline

Every supervised learning experiment in TrueML follows this canonical sequence:

```
┌───────────────────────────────────────────────────────────────┐
│                       TRAINING LOOP                           │
│                                                               │
│  1. FORWARD      y_pred = model.forward(X)                    │
│                  ŷ = Xw + b                                   │
│                                                               │
│  2. LOSS         loss = loss_fn(y, y_pred)                    │
│                  L = mean((y - ŷ)²)                           │
│                                                               │
│  3. GRADIENT     dloss = loss_fn.grad(y, y_pred)              │
│                  dw, db = model.grad(X, dloss)                │
│                  ∂L/∂w = Xᵀ · ∂L/∂ŷ                          │
│                                                               │
│  4. BACKWARD     model.backward(dw, db)                       │
│                  w ← w − η · ∂L/∂w                            │
└───────────────────────────────────────────────────────────────┘
```

There is no `.fit()`. There is no hidden state. You control every step.

---

## Installation

```bash
pip install trueml
```

!!! note "Requirements"
    TrueML requires **Python ≥ 3.13** and depends on `numpy` and `matplotlib`.

---

## Documentation

This documentation is organized using the [Diátaxis](https://diataxis.fr/) framework. Each page serves exactly one purpose.

### :material-school: Tutorials — Learning by doing

Start here if you're new to TrueML. These guided lessons walk you through the core workflow step by step.

| Tutorial | Description |
|----------|-------------|
| [Your First Training Loop](tutorials/your-first-training-loop.md) | Train a linear model from scratch using gradient descent |
| [Comparing Loss Functions](tutorials/comparing-loss-functions.md) | Observe how MSE and MAE loss differ in gradient behavior |

### :material-hammer-wrench: How-to Guides — Accomplishing tasks

Practical guides for specific problems. Use these when you know what you want to do.

| Guide | Description |
|-------|-------------|
| [Manual Gradient Descent](how-to/manual-gradient-descent.md) | Annotated gradient descent walkthrough |
| [Train on Real Data](how-to/train-on-real-data.md) | Apply TrueML to the Housing dataset |
| [Implement Minibatch GD](how-to/implement-minibatch-gd.md) | Extend to minibatch training |
| [Debug Gradient Issues](how-to/debug-gradient-issues.md) | Diagnose vanishing, exploding, and oscillating gradients |

### :material-book-open-page-variant: Reference — The machinery

Complete descriptions of every API component. Consult these when you need an exact specification.

| Module | Components |
|--------|------------|
| **Linear Models** | [LinearRegression](reference/linear-regression.md) · [LogisticRegression](reference/logistic-regression.md) |
| **Loss Functions** | [MSEloss](reference/mse-loss.md) · [MAEloss](reference/mae-loss.md) |
| **Error Functions** | [errors](reference/errors.md) — `residual_error`, `absolute_error` |
| **Activations** | [activations](reference/activations.md) — `sigmoid`, `linear` |
| **Linear Algebra** | [linalg](reference/linalg.md) — `matmul`, `npmatmul` |
| **Visualization** | [visualization](reference/visualization.md) — `plot2d`, `plot3d`, `plot_metrics`, `LivePlot` |
| **Utilities** | [helpers](reference/helpers.md) — `timeit`, `generate`, `memprofile` |

### :material-lightbulb: Explanation — Understanding why

Conceptual discussions that provide context and background. Read these to deepen your understanding.

| Topic | Description |
|-------|-------------|
| [No-Abstraction Philosophy](philosophy.md) | Why zero abstraction matters |
| [About Gradient Descent](explanation/about-gradient-descent.md) | Theory behind the update rule |
| [About Loss Functions](explanation/about-loss-functions.md) | How loss functions shape learning |
| [Calculus Mapping](explanation/calculus-mapping.md) | Chain rule composition in TrueML |

---

## Who TrueML Is For

- **Researchers** who want to read every line of their training loop.
- **Students** learning how gradients actually flow through a linear model.
- **Practitioners** who need a minimal, auditable baseline before layering complexity.

---

## Quick Links

- **New?** Start with the [Your First Training Loop](tutorials/your-first-training-loop.md) tutorial.
- **Curious about the design?** Read the [Philosophy](philosophy.md) page.
- **Looking for a specific function?** Jump to the [API Reference](reference/linear-regression.md).
