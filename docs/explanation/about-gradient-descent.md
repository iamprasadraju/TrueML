# About Gradient Descent

Gradient descent is the iterative optimization algorithm at the core of every training loop in TrueML. This page explains why it works, what the gradient represents, and how the learning rate controls the optimization trajectory.

## Background

The central problem in supervised learning is to find parameters $w, b$ (weights and bias) that minimize a loss function $L(w, b)$ over a dataset. For a linear model $\hat{y} = Xw + b$, the loss measures the discrepancy between predictions $\hat{y}$ and the true targets $y$.

Most loss functions have no closed-form minimizer — there is no formula you can evaluate to get the optimal $w$ and $b$ directly. Gradient descent solves this by turning optimization into a **local search problem**: start somewhere, figure out which direction reduces the loss, take a step, and repeat.

This idea dates back to Cauchy (1847) and is the foundation of nearly all modern machine learning optimization.

## The Core Concept

Gradient descent uses the **first derivative** of the loss with respect to each parameter to decide how to update it. The gradient $\nabla L(w)$ is a vector that points in the direction of steepest *increase* of the loss. To *decrease* the loss, we move in the opposite direction:

$$
w_{t+1} = w_t - \eta \nabla L(w_t)
$$

where $\eta$ (the learning rate) controls the step size.

### Why the Gradient Points Uphill

For a scalar function $f(x)$, the derivative $f'(x)$ gives the slope at $x$: a positive derivative means $f$ increases as $x$ increases. For a vector-valued parameter $w$, the gradient $\nabla L(w)$ is the vector of partial derivatives. By the first-order Taylor expansion:

$$
L(w + \delta) \approx L(w) + \nabla L(w)^\mathsf{T} \delta
$$

To guarantee that $L(w + \delta) < L(w)$ (meaning the loss decreases), we need $\nabla L(w)^\mathsf{T} \delta < 0$. 

The choice $\delta = -\eta \nabla L(w)$ satisfies this inequality (for sufficiently small $\eta$) because:

$$
\nabla L(w)^\mathsf{T} (-\eta \nabla L(w)) = -\eta \|\nabla L(w)\|^2 < 0
$$

This is the fundamental mathematical justification for the gradient descent update rule.

## Chain Rule Composition in TrueML

In TrueML, the gradient computation is intentionally split into two explicit steps to reflect the multivariable chain rule:

1. **Loss gradient** (`loss_fn.grad`): $\frac{\partial L}{\partial \hat{y}}$ — how the loss changes with respect to the prediction.
2. **Parameter gradient** (`model.grad`): $\frac{\partial L}{\partial w} = X^\mathsf{T} \frac{\partial L}{\partial \hat{y}}$ — how the loss changes with respect to the weights.

The separation is deliberate. By keeping these two derivatives distinct, you can inspect each component independently, swap loss functions without changing the model code, and see exactly where the math is happening.

## The Learning Rate and Convergence

The learning rate $\eta$ is the single most important hyperparameter in gradient descent.

- **Too small**: Convergence is slow. The algorithm takes many small steps, wasting computation.
- **Too large**: The update may overshoot the minimum, causing the loss to increase or bounce erratically.
- **Adaptive schemes**: More sophisticated optimizers (Adam, RMSprop) adjust the learning rate per parameter during training. TrueML intentionally omits these to force the user to interact with the raw gradients directly.

### Non-convexity and Local Minima

For linear regression with a convex loss (e.g., `MSEloss`), the loss surface is a convex bowl: any local minimum is a global minimum. Gradient descent is guaranteed to converge to the global optimum given a sufficiently small learning rate.

However, the choice of loss function can introduce non-convexity or non-smoothness. The `MAEloss` is convex but not strictly smooth at zero, which is why it can cause oscillation near the minimum. (See [Comparing Loss Functions](../tutorials/comparing-loss-functions.md)).

## Comparison to Other Optimizers

| Method | Update rule | Gradient used | Key property | TrueML Support |
|--------|-------------|---------------|--------------|----------------|
| Gradient descent | $w \gets w - \eta \nabla L$ | Full batch | Deterministic, stable | Yes (Default) |
| Stochastic GD | $w \gets w - \eta \nabla L_i$ | Single sample | Noisy, fast per step | Yes (via manual loop) |
| Minibatch GD | $w \gets w - \eta \nabla L_{\mathcal{B}}$ | Batch subset | Trade-off noise/cost | Yes (via [Minibatch Guide](../how-to/implement-minibatch-gd.md)) |

All of these follow the same core principle: compute a gradient estimate, scale it by a learning rate, and apply the update.

## Different Perspectives

- **Differential Equations:** Some practitioners view gradient descent as a discretization of a differential equation. The continuous-time limit $\dot{w} = -\nabla L(w)$ describes a gradient flow that converges to a minimum. The discrete steps we take in practice are a finite-difference approximation.
- **Trust Regions:** Others view it through the lens of trust regions: the learning rate defines a radius within which we trust the linear approximation of the loss (from the Taylor expansion). If the step is too large, the linear approximation breaks down, and the loss may increase.

## Further Reading

- **Learn it**: [Your First Training Loop](../tutorials/your-first-training-loop.md) — Walk through gradient descent step by step.
- **Use it**: [How to Implement Minibatch Gradient Descent](../how-to/implement-minibatch-gd.md) — Apply minibatch updates.
- **Details**: [Calculus Mapping](./calculus-mapping.md) — Deep dive into the chain rule composition.
- **Related**: [About Loss Functions](./about-loss-functions.md) — How the choice of loss shapes the gradient landscape.
