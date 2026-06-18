# About the Calculus Mapping

This page explains how the chain rule composes the forward expression of a linear model, the loss function, and the parameter update into the full gradient computation. Understanding this composition is essential for debugging gradients, designing new loss functions, or modifying the training protocol.

## Background

A TrueML training loop consists of four steps: forward pass, loss computation, gradient computation, and parameter update. The gradient computation step is where calculus happens — it is where the derivatives of two different functions (the loss and the linear model) are combined via the chain rule.

The chain rule is the fundamental theorem of differential calculus that describes how to differentiate a composition of functions. If $z = f(g(x))$, then:

$$
\frac{dz}{dx} = \frac{dz}{dg} \cdot \frac{dg}{dx}
$$

In TrueML, the prediction $\hat{y}$ is the output of the linear model ($\hat{y} = Xw + b$), and the loss $L$ is a function of $\hat{y}$. The gradient we need is $\partial L / \partial w$ — how the loss changes with respect to the weights, not the predictions.

## The core concept: two derivatives compose

The gradient computation in TrueML factors into two independent derivatives, composed by the chain rule:

1. **Loss-to-prediction gradient**: $\partial L / \partial \hat{y}$ — computed by `loss_fn.grad()`. This is the gradient of the loss with respect to the model's output. It answers: "if the prediction were slightly different, how would the loss change?"

2. **Prediction-to-parameter gradient**: $\partial \hat{y} / \partial w$ — this is the Jacobian of the linear model. For $\hat{y} = Xw + b$, the Jacobian $\partial \hat{y} / \partial w$ equals $X$, because $\hat{y}$ is linear in $w$.

The chain rule composes them:

$$
\frac{\partial L}{\partial w}
= \frac{1}{n} \left( \frac{\partial \hat{y}}{\partial w} \right)^\mathsf{T}
\frac{\partial L}{\partial \hat{y}}
= \frac{1}{n} X^\mathsf{T}
\frac{\partial L}{\partial \hat{y}}
$$

The factor $1/n$ appears because TrueML computes the mean gradient over all observations, not the sum.

### Why this separation matters

By splitting the gradient into two explicit components, TrueML achieves two things:

- **Replaceability**: The loss-to-prediction gradient depends only on the loss function, not on the model. You can use the same loss function with any model that produces a prediction vector $\hat{y}$.
- **Auditability**: Each component can be inspected independently. If the gradient looks wrong, you can check whether the loss gradient or the model Jacobian is the source.

### The bias gradient

The bias $b$ has Jacobian $\partial \hat{y}_i / \partial b = 1$ for all $i$, so:

$$
\frac{\partial L}{\partial b}
= \frac{1}{n} \sum_{i=1}^{n} \frac{\partial L_i}{\partial \hat{y}_i}
= \text{mean}\left( \frac{\partial L}{\partial \hat{y}} \right)
$$

This is why `loss_fn.grad()` returns a vector $dw$ and a scalar $db$ — the bias gradient is simply the mean of the loss-to-prediction gradient.

## The parameter update is not calculus

The backward step `model.backward(dw, db)` does not involve calculus — it performs a weighted subtraction:

$$
w \gets w - \eta \cdot dw
$$

The gradient $dw$ was already computed; the update is just arithmetic. This separation between gradient computation and parameter update is another deliberate design choice: you can inspect or modify the gradient before applying it.

## Why the chain rule produces the right direction

The gradient $\partial L / \partial w$ computed through this composition is guaranteed to point in the direction of steepest ascent of $L$ with respect to $w$. This is a mathematical consequence of the multivariable chain rule, which is exact for differentiable functions.

For loss functions that are not everywhere differentiable (like AbsoluteError at $r_i = 0$), we use subgradients — a generalization of the gradient that still guarantees descent under standard optimization conditions.

## Further reading

- **Learn it**: [Your First Training Loop](../tutorials/your-first-training-loop.md) — see the calculus in action
- **Use it**: [How to Debug Gradient Issues](../how-to/debug-gradient-issues.md) — practical strategies for gradient problems
- **Details**: [AbsoluteError Reference](../reference/absolute-error.md) — derivative derivation for L1
- **Details**: [SquaredError Reference](../reference/squared-error.md) — derivative derivation for L2
- **Related**: [About Gradient Descent](./about-gradient-descent.md) — theory of the update rule
- **Related**: [About Loss Functions](./about-loss-functions.md) — how loss functions define the gradient
