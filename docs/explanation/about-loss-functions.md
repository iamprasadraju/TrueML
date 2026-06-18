# About Loss Functions

A loss function quantifies the discrepancy between a model's prediction and the ground truth. The choice of loss function fundamentally shapes what the model learns — it determines which errors are penalized, how gradients behave, and what kind of robustness the model exhibits.

## Background

In supervised learning, the loss function is the bridge between the model's output and the optimization algorithm. It serves two roles:

1. **Evaluation**: It assigns a scalar value to each prediction, measuring "how wrong" the model is.
2. **Differentiation**: Its gradient drives the parameter updates that improve the model.

The second role is often overlooked by beginners: the loss function must be differentiable (or admit a subgradient) because gradient descent uses its derivative to decide how to update parameters.

## The core concept: error-to-gradient mapping

Every loss function takes the residual $r_i = y_i - \hat{y}_i$ and produces a per-observation loss $L_i$. The gradient of the loss with respect to the prediction, $\partial L_i / \partial \hat{y}_i$, determines how the model responds to each observation.

TinyMLx provides three loss functions that differ in how they map residuals to gradients:

| Loss | $L_i$ | $\partial L_i / \partial \hat{y}_i$ | Gradient behavior |
|------|-------|-------------------------------------|-------------------|
| SignedError | $r_i$ | $-1$ (constant) | Every observation pulls equally |
| AbsoluteError | $r_i | $-\text{sign}(r_i)$ | Bounded in $[-1, 1]$; robust to outliers |
| SquaredError | $r_i^2$ | $-2 r_i$ | Proportional to error; sensitive to outliers |

### The L1 / L2 tradeoff

The choice between AbsoluteError (L1) and SquaredError (L2) is the most fundamental design decision in loss function selection.

**L1 (AbsoluteError)** treats all residuals with equal weight in the gradient, regardless of magnitude. An observation that is off by 10 units contributes at most $\pm 1$ to the gradient per sample. This makes L1 **robust to outliers** — a single extreme observation cannot dominate the gradient.

**L2 (SquaredError)** penalizes large residuals quadratically. An observation off by 10 units contributes $-20$ to the gradient per sample — 10 times more than an observation off by 1 unit. This makes L2 **sensitive to outliers** but also means the gradient naturally shrinks as the model improves, enabling fine-grained convergence near the optimum.

The tradeoff is well-known in statistics: L1 corresponds to the median of the conditional distribution (less sensitive to extreme values), while L2 corresponds to the mean (fully sensitive to the entire distribution).

### Signed error

The signed error $r_i = y_i - \hat{y}_i$ is not a training loss in the usual sense — it has no gradient method in TinyMLx and is not used for optimization. It exists for diagnostic purposes: inspecting the sign of residuals reveals systematic bias. If all signed errors are positive, the model consistently under-predicts.

## Why TinyMLx separates loss from gradient

In most frameworks, the loss function and its gradient are combined. PyTorch's `MSELoss` computes both the loss value and the gradient in a single backward pass. TinyMLx separates them:

```python
error = loss_fn(y, y_pred)        # compute loss value
dw, db = loss_fn.grad(X, error)   # compute gradient separately
```

This separation is deliberate:
- You can inspect the error vector before computing gradients.
- You can modify the error (clip, reweight, mask) before passing it to `.grad()`.
- The gradient formula is explicit and auditable.

## Different perspectives

Some practitioners view loss functions as defining a **probabilistic model**. Minimizing squared error is equivalent to maximum likelihood estimation under a Gaussian noise model, while minimizing absolute error corresponds to maximum likelihood under a Laplacian noise model. From this perspective, the loss function encodes assumptions about the data-generating distribution.

Others view loss functions through the lens of **robust statistics**: L1 is a robust estimator (its influence function is bounded), while L2 is not (a single outlier can have unbounded influence on the estimate).

In the optimization community, the focus is on **convexity** and **smoothness**: L2 is smooth (infinitely differentiable) while L1 is non-smooth (subdifferentiable at zero), which affects which optimization algorithms can be applied and how quickly they converge.

## Further reading

- **Learn it**: [Comparing Loss Functions](../tutorials/comparing-loss-functions.md) — hands-on tutorial comparing L1 vs L2 behavior
- **Use it**: [How to Train on a Real Dataset](../how-to/train-on-real-data.md) — apply loss functions to real data
- **Details**: [AbsoluteError Reference](../reference/absolute-error.md) — mathematical contract for L1
- **Details**: [SquaredError Reference](../reference/squared-error.md) — mathematical contract for L2
- **Details**: [SignedError Reference](../reference/signed-error.md) — mathematical contract for signed error
- **Related**: [About Gradient Descent](./about-gradient-descent.md) — how gradients drive optimization
