# About Gradient Descent

Gradient descent is the iterative optimization algorithm at the core of every training loop in TrueML. This page explains why it works, what the gradient represents, and how the learning rate controls the optimization trajectory.

## Background

The central problem in supervised learning is to find parameters $\theta$ (weights and bias) that minimize a loss function $\mathcal{L}(\theta)$ over a dataset. For a linear model $\hat{y} = Xw + b$, the loss measures the discrepancy between predictions $\hat{y}$ and the true targets $y$.

Most loss functions have no closed-form minimizer — there is no formula you can evaluate to get the optimal $w$ and $b$ directly. Gradient descent solves this by turning optimization into a **local search problem**: start somewhere, figure out which direction reduces the loss, take a step, and repeat.

This idea dates back to Cauchy (1847) and is the foundation of nearly all modern deep learning optimization.

## The core concept

Gradient descent uses the **first derivative** of the loss with respect to each parameter to decide how to update it. The gradient $\nabla \mathcal{L}(\theta)$ is a vector that points in the direction of steepest *increase* of the loss. To *decrease* the loss, we move in the opposite direction:

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)
$$

where $\eta$ (the learning rate) controls the step size.

### Why the gradient points uphill

For a scalar function $f(x)$, the derivative $f'(x)$ gives the slope at $x$: a positive derivative means $f$ increases as $x$ increases. For a vector-valued parameter $\theta$, the gradient $\nabla \mathcal{L}(\theta)$ is the vector of partial derivatives. By the first-order Taylor expansion:

$$
\mathcal{L}(\theta + \delta) \approx \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\mathsf{T} \delta
$$

To guarantee that $\mathcal{L}(\theta + \delta) < \mathcal{L}(\theta)$, we want $\nabla \mathcal{L}(\theta)^\mathsf{T} \delta < 0$. The choice $\delta = -\eta \nabla \mathcal{L}(\theta)$ satisfies this inequality (for sufficiently small $\eta$) because:

$$
\nabla \mathcal{L}(\theta)^\mathsf{T} (-\eta \nabla \mathcal{L}(\theta)) = -\eta \|\nabla \mathcal{L}(\theta)\|^2 < 0
$$

This is the fundamental justification for the gradient descent update rule.

### The chain rule connects model and loss

In TrueML, the gradient computation is split into two explicit steps:

1. **Loss gradient**: $\partial L / \partial \hat{y}$ — how the loss changes with respect to the prediction.
2. **Parameter gradient**: $\partial L / \partial w = \frac{1}{n} X^\mathsf{T} (\partial L / \partial \hat{y})$ — how the loss changes with respect to the weights, via the chain rule through the linear model.

The separation is deliberate. By keeping these two derivatives distinct, the user can inspect each component independently.

## The learning rate and convergence

The learning rate $\eta$ is the single most important hyperparameter in gradient descent.

- **Too small**: Convergence is slow. The algorithm takes many small steps, wasting computation.
- **Too large**: The update may overshoot the minimum, causing the loss to increase (diverge).
- **Adaptive schemes**: More sophisticated optimizers (Adam, RMSprop) adjust the learning rate per parameter during training. TrueML intentionally omits these — the user implements the update rule manually.

### Non-convexity and local minima

For linear regression with a convex loss (e.g., squared error), the loss surface is convex: any local minimum is a global minimum. Gradient descent is guaranteed to converge to the global optimum given a sufficiently small learning rate.

However, the choice of loss function can introduce non-convexity. The absolute error loss is convex but not strictly convex — its minimum is a set rather than a single point, which is why L1-trained models can have multiple equally-good parameter configurations.

## Comparison to other optimizers

| Method | Update rule | Gradient used | Key property |
|--------|-------------|---------------|--------------|
| Gradient descent | $\theta \gets \theta - \eta \nabla \mathcal{L}$ | Full batch | Deterministic, stable |
| Stochastic GD | $\theta \gets \theta - \eta \nabla \mathcal{L}_i$ | Single sample | Noisy, fast per step |
| Minibatch GD | $\theta \gets \theta - \eta \nabla \mathcal{L}_{\mathcal{B}}$ | Batch subset | Trade-off between noise and cost |

All of these follow the same core principle: compute a gradient estimate, scale it by a learning rate, and apply the update.

## Different perspectives

Some practitioners view gradient descent as a **discretization of a differential equation**. The continuous-time limit $\dot{\theta} = -\nabla \mathcal{L}(\theta)$ describes a gradient flow that converges to a minimum. The discrete steps we take in practice are a finite-difference approximation.

Others view it through the lens of **trust regions**: the learning rate defines a radius within which we trust the linear approximation of the loss. If the step is too large, the approximation breaks and the loss may increase.

## Further reading

- **Learn it**: [Your First Training Loop](../tutorials/your-first-training-loop.md) — walk through gradient descent step by step
- **Use it**: [How to Implement Minibatch Gradient Descent](../how-to/implement-minibatch-gd.md) — apply minibatch updates
- **Details**: [Calculus Mapping](./calculus-mapping.md) — chain rule composition for all TrueML operations
- **Related**: [About Loss Functions](./about-loss-functions.md) — how the choice of loss changes the gradient landscape
