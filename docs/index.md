# tinyMLx Documentation

**Version 0.1.0** — A no-abstraction mathematical sandbox for machine learning research.

---

## The No-Abstraction Philosophy

Most machine learning frameworks present a **black-box contract**: data goes in, a trained model comes out. `sklearn.linear_model.LinearRegression().fit(X, y)` conceals the forward pass, the loss evaluation, the gradient computation, and the parameter update behind a single method call. While convenient for production, this opacity is antithetical to **understanding**.

TinyMLx adopts the opposite stance: **zero abstraction**. Every mathematical operation in the learning pipeline is a first-class function you invoke explicitly. The library provides the *primitive operations*; you write the *protocol*.

### The Four-Step Pipeline

Every supervised learning experiment in TinyMLx follows this canonical sequence:

```
┌─────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                        │
│                                                         │
│  1. FORWARD     y_pred = model.forward(X)               │
│       ŷ = Xw + b                                        │
│                                                         │
│  2. LOSS        error = loss_fn(y, y_pred)              │
│       L = |y – ŷ|                                       │
│                                                         │
│  3. GRADIENT    dw, db = loss_fn.grad(X, error)         │
│       ∂L/∂w = (1/n) Xᵀ · ∂L/∂ŷ                         │
│                                                         │
│  4. BACKWARD      model.backward(dw, db)                │
│       w ← w – η · ∂L/∂w                                 │
└─────────────────────────────────────────────────────────┘
```

There is no `.fit()`. There is no hidden state. The user controls every step.

---

## Documentation by Category

This documentation is organized by the Diátaxis framework: each document serves exactly one user need.

### Tutorials — learning by doing

Start here if you're new to TinyMLx. These guided lessons walk you through the core workflow step by step.

- [Your First Training Loop](tutorials/your-first-training-loop.md) — train a linear model from scratch
- [Comparing Loss Functions](tutorials/comparing-loss-functions.md) — observe how L1 and L2 loss differ

### How-to Guides — accomplishing tasks

Practical guides for specific problems. Use these when you know what you want to do.

- [Manual Gradient Descent](how-to/manual-gradient-descent.md) — annotated gradient descent walkthrough
- [Train on Real Data](how-to/train-on-real-data.md) — apply TinyMLx to Housing.csv
- [Implement Minibatch GD](how-to/implement-minibatch-gd.md) — extend to minibatch training

### Reference — the machinery

Complete descriptions of every API component. Consult these when you need an exact specification.

- **LinearModels**: [LinearRegression](reference/linear-regression.md), [LogisticRegression](reference/logistic-regression.md)
- **Loss Functions**: [AbsoluteError](reference/absolute-error.md), [SquaredError](reference/squared-error.md), [SignedError](reference/signed-error.md)
- **Operations**: [gemm — Matrix Operations](reference/gemm.md), [helpers — Utilities](reference/helpers.md)

### Explanation — understanding why

Conceptual discussions that provide context and background. Read these when you want to deepen your understanding.

- [No-Abstraction Philosophy](explanation/no-abstraction-philosophy.md) — why zero abstraction matters
- [About Gradient Descent](explanation/about-gradient-descent.md) — theory behind the update rule
- [About Loss Functions](explanation/about-loss-functions.md) — how loss functions shape learning
- [Calculus Mapping](explanation/calculus-mapping.md) — chain rule composition

---

## Who This Library Is For

- Researchers who want to **read every line** of their training loop.
- Students learning how gradients actually **flow** through a linear model.
- Practitioners who need a **minimal, auditable** baseline before layering complexity.
<<<<<<< Updated upstream

---

**Next:** Read the [Philosophy](philosophy.md) for an extended discussion of why abstraction-free design matters, or jump to the [API Reference](api/models/linear_regression.md) for the mathematical contract of each class.
=======
>>>>>>> Stashed changes
