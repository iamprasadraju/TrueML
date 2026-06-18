# How to Debug Gradient Issues

This guide shows you how to diagnose and fix common gradient problems in TrueML training loops.

## When to use this guide

Use this when your model's loss is not decreasing, diverges to infinity, oscillates without converging, or converges to a suboptimal solution.

## Before you start

- You have a working training loop (see [Manual Gradient Descent](./manual-gradient-descent.md))
- Familiarity with the [Calculus Mapping](../explanation/calculus-mapping.md) explanation of chain rule composition

## Context

Gradient descent is a simple algorithm, but small problems in the gradient computation can produce large failures. Most issues fall into one of four categories: vanishing gradients, exploding gradients, learning rate sensitivity, or subgradient ambiguity.

## Steps

### 1. Identify the symptom

| Symptom | Likely cause |
|---------|-------------|
| Loss increases every epoch | Learning rate too high |
| Loss decreases then plateaus far from zero | Learning rate too low |
| Loss oscillates without converging | L1 loss with insufficient LR decay |
| Loss jumps to NaN | Exploding gradient (L2 with outlier) |
| Loss decreases but model parameters are wrong | Feature scale imbalance |

### 2. Check feature standardization

Gradient direction is sensitive to feature scale. If one feature (e.g., area in square feet) has values in the thousands and another (e.g., number of bedrooms) has single-digit values, the gradient will be dominated by the larger-scale feature.

```python
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
```

After standardization, each feature contributes proportionally to the gradient.

### 3. Inspect the gradient values

Print the gradient before calling `backward`:

```python
y_pred = model.forward(X)
error = loss_fn(y, y_pred)
dw, db = loss_fn.grad(X, error)

print("dw norm:", np.linalg.norm(dw))
print("db:", db)
```

- If `dw norm` is > 10, the gradient is large — reduce learning rate.
- If `dw norm` is < 1e-8, the gradient has vanished — increase learning rate or switch to SquaredError.
- If `dw norm` oscillates without trending downward, you may need learning rate scheduling.

### 4. Handle vanishing gradients with L1 loss

AbsoluteError produces a gradient of constant magnitude $\pm 1$ per observation. Near the optimum, the gradient does not shrink — the model takes the same-size step regardless of how close it is to the minimum. This can cause oscillation.

**Solution:** Decrease the learning rate or switch to SquaredError:

```python
from trueml.lossfunc import SquaredError
loss_fn = SquaredError()
```

With SquaredError, the gradient magnitude is proportional to the error, so it naturally shrinks near the optimum.

### 5. Handle exploding gradients with L2 loss

SquaredError produces gradients proportional to the error. A single outlier with $|y - \hat{y}| > 10$ produces a gradient component of magnitude $2|e_i| > 20$, which can dominate the update.

**Solution:** Clip the gradient before calling `backward`:

```python
dw = np.clip(dw, -1.0, 1.0)
db = np.clip(db, -1.0, 1.0)
```

Or switch to AbsoluteError, which is robust to outliers.

### 6. Adjust the learning rate

If the loss increases or oscillates, reduce `lr` by a factor of 10:

```python
model = LinearRegression(n_features=d, lr=0.001)  # instead of 0.01
```

If the loss decreases very slowly, increase `lr`:

```python
model = LinearRegression(n_features=d, lr=0.1)  # instead of 0.01
```

A good heuristic: start with `lr = 0.01` for standardized data, `lr = 0.001` for unstandardized data with large feature values.

## Troubleshooting

**Problem: Loss goes to NaN after a few epochs**
Solution: This is almost always an exploding gradient. Clip gradients or reduce learning rate.

**Problem: Loss is flat from epoch 1**
Solution: The learning rate is too low or the gradient is zero. Check that `dw` is non-zero. If using L1 loss, check that not all predictions equal the targets (subgradient of zero).

**Problem: Training is slow even though gradients look reasonable**
Solution: Try minibatch gradient descent (see [Implement Minibatch GD](./implement-minibatch-gd.md)) for faster per-epoch progress, or increase the learning rate.

## Related guides

- [Manual Gradient Descent](./manual-gradient-descent.md) — baseline training loop
- [How to Train on a Real Dataset](./train-on-real-data.md) — feature standardization in practice
- [About Gradient Descent](../explanation/about-gradient-descent.md) — theory of learning rate and convergence
- [About Loss Functions](../explanation/about-loss-functions.md) — L1 vs L2 gradient behavior
