# Philosophy

## Why "No Abstraction"?

### The Problem with Hidden `fit()`

Consider the canonical scikit-learn workflow:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

What happens inside `.fit()`? For a linear model:

1. A solver is chosen (often by implicit heuristic).
2. The normal equations $w = (X^\mathsf{T}X)^{-1}X^\mathsf{T}y$ are solved, or an iterative method (SGD, SVD) is invoked.
3. Residuals are computed, but never exposed.
4. Coefficients are stored as `model.coef_`.

Each of these steps encodes a **mathematical decision** — and each decision is hidden from the caller. TrueML makes every decision explicit because **in research, the decision is the result**.

### The Laboratory Manual Metaphor

This documentation is styled as a **Laboratory Manual**, not a product reference. A laboratory manual does not tell you "press the green button." It tells you:

- What reagents you need (inputs).
- What reaction to expect (forward pass).
- How to measure the yield (loss).
- How to purify the product (gradient).
- How to adjust the procedure (update).

Each [API Reference](api/models/linear_regression.md) page in this manual is structured as a **Mathematical Contract** — a precise statement of inputs, outputs, and derivatives. The contracts are designed to be read alongside the source code: the implementation is the math.

### Statelessness Explained

A model in TrueML has **two kinds of attributes**:

| Kind | Examples | Persists across calls? |
|------|----------|-----------------------|
| Parameters | `weights`, `bias` | Yes — these are what training changes |
| Hyper-parameters | `lr` | Yes — these control how training proceeds |
| Data cache | — | **Never** — no `X_train_`, no `y_train_` |

The absence of data caching is deliberate. If you want to compute gradients, you must supply both the design matrix $X$ and the error vector $e$ to `.grad()`. This forces you to **prove** that you know where your gradients came from.

### Comparison with Other Frameworks

| Framework | Training API | Gradient visibility | User controls update? |
|-----------|-------------|--------------------|-----------------------|
| scikit-learn | `fit(X, y)` | None | No |
| PyTorch | `loss.backward()` | `param.grad` | Yes (via `optimizer.step()`) |
| TrueML | Explicit loop | Returned from `grad()` | Yes (via `backward(dw, db)`) |

TrueML goes beyond PyTorch in transparency: there is no autograd tape, no implicit graph building. The user calls `grad()` and receives **numpy arrays** containing the partial derivatives. What you do with them is your own procedure.

### When (Not) to Use TrueML

**Use when:** You are learning or teaching ML fundamentals, prototyping a new optimization algorithm, or need a maximally transparent baseline.

**Prefer other tools when:** You need GPU acceleration, autograd for deep networks, or production-grade solvers (QR decomposition, L-BFGS).

TrueML is a **sandbox**, not a fortress. It is designed to be read, modified, and outgrown.
