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

### Statelessness as a Design Principle

TinyMLx models are **stateless with respect to data**. They hold *parameters* (weights, bias) but never cache a training example. This means:

- **No accidental leakage**: a model cannot remember a previous batch.
- **Explicit dataflow**: you always know what data produced what gradient.
- **Composability**: any step can be replaced, inspected, or debugged in isolation.

If you want to log gradients before the update, you print them. If you want to try a custom update rule, you write it yourself. The library does not abstract away what you need to see.

### Who This Library Is For

- Researchers who want to **read every line** of their training loop.
- Students learning how gradients actually **flow** through a linear model.
- Practitioners who need a **minimal, auditable** baseline before layering complexity.

---

**Next:** Read the [Philosophy](philosophy.md) for an extended discussion of why abstraction-free design matters, or jump to the [API Reference](api/models/linear_regression.md) for the mathematical contract of each class.
