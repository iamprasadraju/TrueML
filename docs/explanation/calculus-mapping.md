# About the Calculus Mapping

This page explains how the chain rule mathematically composes the forward expression of a model, the loss function, and the parameter update into the full gradient computation. 

Understanding this composition is essential for debugging gradients, designing custom loss functions, or modifying the TrueML training protocol.

## Background

A TrueML training loop consists of four steps: Forward pass, Loss computation, Gradient computation, and Backward update. 

The **Gradient computation** step is where the calculus happens. It is where the derivatives of two completely different functions (the loss function and the linear model) are stitched together using the chain rule.

The chain rule is the fundamental theorem of differential calculus for composite functions. If $z = f(g(x))$, then:
$$
\frac{dz}{dx} = \frac{dz}{dg} \cdot \frac{dg}{dx}
$$

In our machine learning context:
- $w$ is the input (weights).
- $g$ is the model, producing predictions $\hat{y}$.
- $f$ is the loss function, producing the scalar loss $L$.
- We need $\frac{\partial L}{\partial w}$ to update the weights.

## Two Derivatives Compose

In TrueML, the gradient computation is intentionally broken into two method calls corresponding to the two parts of the chain rule:

1. **Loss-to-prediction gradient (`loss_fn.grad`)**: 
   $$ \frac{\partial L}{\partial \hat{y}} $$
   This asks: *"If the prediction were slightly different, how would the loss change?"*

2. **Prediction-to-parameter Jacobian (`model.grad`)**: 
   $$ \frac{\partial \hat{y}}{\partial w} $$
   This asks: *"If the weights were slightly different, how would the prediction change?"*

### The Mathematical Derivation

Let our linear model be defined as:
$$ \hat{y} = Xw + b $$

The Jacobian of the predictions with respect to the weights is simply the design matrix:
$$ \frac{\partial \hat{y}}{\partial w} = X $$

By the multivariable chain rule, we compose the two parts. TrueML computes the average gradient across $n$ observations, introducing a $\frac{1}{n}$ factor (though often this factor is baked into the loss gradient itself). 

Assuming the loss gradient $\frac{\partial L}{\partial \hat{y}}$ is a vector of shape $(n, 1)$, the full gradient with respect to $w$ is computed via a dot product:

$$
\frac{\partial L}{\partial w} = X^\mathsf{T} \frac{\partial L}{\partial \hat{y}}
$$

If you look at the source code for `LinearRegression.grad()`, you will see this exact equation:
```python
def grad(self, X: np.ndarray, loss_gradient: np.ndarray):
    dw = X.T @ loss_gradient
    # ...
```

### The Bias Gradient

The bias $b$ is added uniformly to every prediction. Therefore, its Jacobian is a vector of ones:
$$ \frac{\partial \hat{y}_i}{\partial b} = 1 \quad \text{for all } i $$

By the chain rule, the gradient of the loss with respect to the bias is simply the sum of the loss gradients:
$$
\frac{\partial L}{\partial b} = \sum_{i=1}^{n} \frac{\partial L_i}{\partial \hat{y}_i}
$$

*(Again, if the loss gradient is pre-scaled by $1/n$, this becomes a mean).*

In the source code, this is implemented as:
```python
db = np.sum(loss_gradient)
```

## Why This Separation Matters

By forcing you to call `loss_fn.grad` and pass the result into `model.grad`, TrueML achieves two architectural goals:

1. **Replaceability**: The loss-to-prediction gradient depends *only* on the loss function. The model doesn't care if you used `MSEloss` or `MAEloss`. As long as you provide an array of shape $(n,)$, the matrix math in `model.grad` remains identical.
2. **Auditability**: If a gradient is exploding, you can print the output of `loss_fn.grad()` and the output of `model.grad()` independently to isolate whether the math failed in the loss derivative or in the model Jacobian.

## The Parameter Update is NOT Calculus

The fourth step in the pipeline, `model.backward(dw, db)`, contains no calculus. 

It simply performs a weighted subtraction:
$$ w \gets w - \eta \cdot dw $$

The calculus finished the moment `dw` was created. This separation proves that gradient *computation* (calculus) and parameter *optimization* (arithmetic) are two fundamentally different operations, even though frameworks like PyTorch blur them together inside `optimizer.step()`.

## Further Reading
- [LinearRegression Reference](../reference/linear_model/linear_regression.md) — See the mathematical contract implemented in code.
- [About Gradient Descent](about-gradient-descent.md) — Why the gradient vector points toward the steepest ascent.
