# About Loss Functions

A loss function quantifies the discrepancy between a model's prediction and the ground truth. The choice of loss function fundamentally shapes what the model learns — it determines which errors are penalized, how gradients behave, and what kind of robustness the model exhibits.

## Background

In supervised learning, the loss function is the bridge between the model's output and the optimization algorithm. It serves two distinct roles:

1. **Evaluation**: It assigns a scalar value to each prediction, measuring "how wrong" the model is.
2. **Differentiation**: Its gradient drives the parameter updates that improve the model.

The second role is critical: the loss function must be differentiable (or admit a subgradient) because gradient descent uses its derivative to dictate how parameters should change.

## Error-to-Gradient Mapping

Every loss function takes the raw residual error $(y_i - \hat{y}_i)$ and produces a scalar loss $L_i$. The gradient of the loss with respect to the prediction, $\frac{\partial L_i}{\partial \hat{y}_i}$, determines how strongly the model responds to that specific observation.

TrueML provides different loss and error components that illustrate this mapping:

| Component | Class/Function | Formula | Gradient $\frac{\partial L_i}{\partial \hat{y}_i}$ | Gradient Behavior |
|-----------|----------------|---------|--------------------------------|-------------------|
| Signed Error | `residual_error` | $y_i - \hat{y}_i$ | N/A (Not a loss) | Used for diagnostics, not training |
| L1 Loss | `MAEloss` | $\|y_i - \hat{y}_i\|$ | $\text{sign}(\hat{y}_i - y_i)$ | Bounded in $[-1, 1]$; robust to outliers |
| L2 Loss | `MSEloss` | $(y_i - \hat{y}_i)^2$ | $2(\hat{y}_i - y_i)$ | Proportional to error; sensitive to outliers |

*(Note: TrueML's `MSEloss.grad` scales the gradient by $1/n$ automatically, yielding $\frac{2}{n}(\hat{y} - y)$).*

## The L1 / L2 Tradeoff

The choice between `MAEloss` (L1) and `MSEloss` (L2) is a fundamental design decision.

**L1 (`MAEloss`)** treats all residuals with equal weight in the gradient, regardless of magnitude. An observation that is off by 10 units contributes at most $\pm 1$ to the gradient. This makes L1 **robust to outliers** — a single extreme observation cannot dominate the gradient update.

**L2 (`MSEloss`)** penalizes large residuals quadratically. An observation off by 10 units contributes proportionally more to the gradient than an observation off by 1 unit. This makes L2 **highly sensitive to outliers**. However, it also ensures the gradient naturally shrinks as the model improves, enabling smooth, fine-grained convergence near the optimum.

### The Diagnostic Role of Signed Error

The `residual_error` function in the `errors` module is not a training loss — it has no `grad()` method and is not used for optimization. It exists purely for diagnostic purposes. Inspecting the sign of residuals reveals systematic bias. For example, if all residual errors are positive, your model is consistently under-predicting the targets.

## Why TrueML Separates Loss from Gradient

In most high-level frameworks (like PyTorch), the loss value and the gradient are computed together implicitly (e.g., calling `.backward()` on a loss scalar). TrueML forces you to separate them:

```python
# 1. Evaluate "how wrong" we are (Evaluation role)
loss_value = loss_fn(y_true, y_pred)

# 2. Compute the derivative (Differentiation role)
loss_grad = loss_fn.grad(y_true, y_pred)
```

This separation is deliberate. It allows you to:
- Inspect the gradient vector `loss_grad` *before* it is passed to the model.
- Modify the gradient (e.g., apply a custom mask or reweighting) before applying the chain rule.
- See exactly where the calculus enters the code.

## Theoretical Perspectives

Different fields view loss functions through different theoretical lenses:

### 1. Probabilistic Perspective (Maximum Likelihood)
Minimizing `MSEloss` is mathematically equivalent to Maximum Likelihood Estimation (MLE) assuming the data was generated with Gaussian (Normally distributed) noise. Minimizing `MAEloss` is equivalent to MLE assuming Laplacian noise (which has fatter tails, mathematically explaining its robustness to outliers).

### 2. Robust Statistics Perspective
In robust statistics, an estimator is analyzed by its "influence function" — how much a single outlier can change the estimate. `MAEloss` is a robust estimator because its influence function is bounded (it maxes out at $\pm 1$). `MSEloss` is not robust because a single infinite outlier can exert infinite influence on the gradient.

### 3. Optimization Perspective
In optimization theory, `MSEloss` is prized because it is smooth and infinitely differentiable, making it highly amenable to second-order methods (like Newton's method) and guaranteeing fast convergence near the minimum. `MAEloss` is non-smooth (it has a sharp "V" shape at zero), meaning it only has a subgradient at zero, which can cause gradient descent algorithms to oscillate indefinitely around the minimum without learning rate decay.

## Further Reading

- **Tutorial**: [Comparing Loss Functions](../tutorials/comparing-loss-functions.md) — A hands-on demonstration of L1 vs L2 robustness.
- **Reference**: [MSEloss](../reference/losses/mean_squared_error.md) and [MAEloss](../reference/losses/mean_absolute_error.md).
- **Related**: [Calculus Mapping](./calculus-mapping.md) — How the loss gradient combines with the model gradient.
