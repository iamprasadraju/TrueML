# Philosophy

## Machine Learning Without Hidden Abstractions

TrueML adopts a radically transparent approach to machine learning: **every mathematical operation in the learning pipeline is a first-class function you invoke explicitly**. The library provides the *primitive operations*; you write the *protocol*.

---

## The Problem with Hidden `.fit()`

Consider the canonical scikit-learn workflow:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

What happens inside `.fit()`?

1. A solver is chosen — often by implicit heuristic based on data characteristics.
2. The normal equations $w = (X^\mathsf{T}X)^{-1}X^\mathsf{T}y$ are solved, or an iterative method (SGD, SVD) is invoked.
3. Residuals are computed, but never exposed.
4. Coefficients are stored as `model.coef_`.

Each of these steps encodes a **mathematical decision** — and each decision is hidden from the caller.

!!! warning "The hidden cost of convenience"
    When the solver, the loss function, and the update rule are all implicit, debugging becomes archaeology. You cannot inspect what you cannot see.

TrueML makes every decision explicit because **in research, the decision *is* the result**.

---

## The Laboratory Manual Metaphor

This documentation is styled as a **Laboratory Manual**, not a product reference. A laboratory manual does not tell you "press the green button." It tells you:

| Lab Step | ML Equivalent | TrueML API |
|----------|--------------|------------|
| Prepare reagents | Load and preprocess data | `X`, `y` as NumPy arrays |
| Run the reaction | Forward pass | `model.forward(X)` |
| Measure the yield | Compute loss | `loss_fn(y, y_pred)` |
| Analyze the residuals | Compute gradient | `loss_fn.grad(y, y_pred)` + `model.grad(X, dloss)` |
| Adjust the procedure | Update parameters | `model.backward(dw, db)` |

Each [API Reference](reference/linear_model/linear_regression.md) page is structured as a **Mathematical Contract** — a precise statement of inputs, outputs, and derivatives. The contracts are designed to be read alongside the source code: **the implementation is the math**.

---

## Statelessness by Design

A model in TrueML has exactly two kinds of attributes:

| Kind | Examples | Persists across calls? |
|------|----------|----------------------|
| **Parameters** | `weights`, `bias` | Yes — these are what training changes |
| **Hyperparameters** | `lr` | Yes — these control how training proceeds |
| **Data cache** | — | **Never** — no `X_train_`, no `y_train_` |

The absence of data caching is deliberate. If you want to compute gradients, you must supply both the design matrix $X$ and the loss gradient to `model.grad()`. This forces you to **prove** that you know where your gradients came from.

!!! tip "Statelessness enables composability"
    Because models hold no data references, the same model instance can be used with different datasets, different loss functions, or different training protocols without any hidden interactions.

---

## Comparison with Other Frameworks

| | scikit-learn | PyTorch | TrueML |
|---|---|---|---|
| **Training API** | `fit(X, y)` | `loss.backward()` | Explicit loop |
| **Gradient visibility** | None | `param.grad` | Returned from `model.grad()` |
| **User controls update?** | No | Yes (via `optimizer.step()`) | Yes (via `model.backward(dw, db)`) |
| **Autograd tape?** | N/A | Yes (implicit graph) | No — manual chain rule |
| **Data cached in model?** | Yes (`X_train_`, etc.) | No | No |

TrueML goes beyond PyTorch in transparency: there is no autograd tape, no implicit graph building. The user calls `model.grad()` and receives **NumPy arrays** containing the partial derivatives. What you do with them is entirely your decision.

---

## The Separation Principle

TrueML separates every concern into its own module:

```
trueml/
├── linearmodel/     # Models: parameters + forward/grad/backward
├── losses/          # Loss functions: scalar loss + gradient w.r.t. predictions
├── errors/          # Raw error computations: residual, absolute
├── activations/     # Pure element-wise functions: sigmoid, linear
├── linalg/          # Linear algebra primitives: matmul
├── viz/             # Matplotlib-based plotting
├── visualize/       # Plotly-based interactive plotting
└── helpers/         # Benchmarking decorators: timeit, memprofile
```

This separation is not organizational convention — it is a **design constraint**. Models do not know about loss functions. Loss functions do not know about models. The training loop is the user's code that connects them.

---

## When (Not) to Use TrueML

**Use TrueML when:**

- You are **learning** or **teaching** ML fundamentals
- You are **prototyping** a new optimization algorithm
- You need a **minimal, auditable baseline** before layering complexity
- You want to **understand** what happens between `.fit()` and `model.coef_`

**Prefer other tools when:**

- You need GPU acceleration
- You need autograd for deep networks
- You need production-grade solvers (QR decomposition, L-BFGS)
- You need a model zoo with pretrained weights

!!! quote "Design philosophy"
    TrueML is a **sandbox**, not a fortress. It is designed to be read, modified, and outgrown.
