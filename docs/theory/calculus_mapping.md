# Calculus Mapping

A reference for how each library operation maps to its forward expression, its gradient, and the chain rule that connects them.

---

## Derivative Reference Table

| Step | Code | Forward Expression | Gradient Expression |
|------|------|-------------------|-------------------|
| Predict | `model.forward(X)` | $\hat{y} = Xw + b$ | — |
| L1 Loss | `loss(y, ŷ)` | $L_i = \|y_i - \hat{y}_i\|$ | $\displaystyle \frac{\partial L_i}{\partial \hat{y}_i} = -\text{sign}(y_i - \hat{y}_i)$ |
| L2 Loss | `loss(y, ŷ)` | $L_i = (y_i - \hat{y}_i)^2$ | $\displaystyle \frac{\partial L_i}{\partial \hat{y}_i} = -2(y_i - \hat{y}_i)$ |
| Differentiate | `loss.grad(X, e)` | — | $\displaystyle \frac{\partial L}{\partial w} = \frac{1}{n} X^\mathsf{T} \frac{\partial L}{\partial \hat{y}}$  <br> $\displaystyle \frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^n \frac{\partial L_i}{\partial \hat{y}_i}$ |
| Update | `model.backward(dw, db)` | — | $w \gets w - \eta \cdot \frac{\partial L}{\partial w}$ <br> $b \gets b - \eta \cdot \frac{\partial L}{\partial b}$ |

---

## Chain Rule Composition

The full gradient computation composes three derivatives via the chain rule:

$$
\frac{\partial L}{\partial w}
= \frac{1}{n} X^\mathsf{T}
\;
\underbrace{
\frac{\partial L}{\partial \hat{y}}
}_{
\text{from loss.grad()}
}
=
\frac{1}{n} X^\mathsf{T}
\;
\underbrace{
\Bigl(
\frac{\partial \hat{y}}{\partial w}
\Bigr)^\mathsf{T}
}_{
= X
}
\;
\underbrace{
\frac{\partial L}{\partial \hat{y}}
}_{
\text{from loss function}
}
$$

Because $\hat{y} = Xw + b$, the Jacobian $\partial \hat{y} / \partial w = X$, hence the transpose in the formula.

---

## Choosing a Loss Function

| Criterion | AbsoluteError (L1) | SquaredError (L2) |
|-----------|-------------------|-------------------|
| Outlier robustness | High | Low |
| Gradient scale | $\pm 1$ (bounded) | $\propto |error|$ (unbounded) |
| Smoothness | Not differentiable at 0 | Smooth everywhere |
| Convergence behavior | Constant step in parameter space | Step shrinks as error shrinks |

### Troubleshooting Common Gradient Pitfalls

#### 1. Vanishing Gradients with AbsoluteError

When $y_i \approx \hat{y}_i$, the error is small but $\partial L_i / \partial \hat{y}_i = \pm 1$ — the gradient **does not vanish** as the optimum is approached. This can cause oscillation around the minimum because the update step size does not decay with the error.

**Mitigation:** Decrease the learning rate $\eta$ or switch to SquaredError, whose gradient $\propto (y - \hat{y})$ naturally shrinks near the optimum.

#### 2. Exploding Gradients with SquaredError

A single outlier with $|y_i - \hat{y}_i| \gg 1$ produces a gradient of magnitude $2|e_i|$, which can cause the parameter update to overshoot.

**Mitigation:** Clip gradients before calling `backward`:

```python
dw = np.clip(dw, -1.0, 1.0)
db = np.clip(db, -1.0, 1.0)
```

#### 3. Learning Rate Sensitivity

The optimal learning rate depends on the scale of $X$. If features have very different magnitudes, the gradient direction $dw = \frac{1}{n} X^\mathsf{T} g$ will be dominated by large-magnitude features.

**Mitigation:** Standardize $X$ to zero mean and unit variance before training:

```python
X = (X - X.mean(axis=0)) / X.std(axis=0)
```

This ensures each feature contributes proportionally to the gradient.

#### 4. Subgradient Ambiguity at Zero

`AbsoluteError.grad` uses `np.sign(0) = 0` as the subgradient at the non-differentiable point $y_i = \hat{y}_i$. This is a valid choice (the subdifferential of $|x|$ at $x=0$ is $[-1, 1]$), but it means that exactly correct predictions contribute zero gradient, which can slow convergence when many predictions are exactly correct.

**Mitigation:** This is rarely an issue in practice due to floating-point precision; exact equality is extremely unlikely.
