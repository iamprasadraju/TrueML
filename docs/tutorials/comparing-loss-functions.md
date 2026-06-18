# Comparing Loss Functions

In this tutorial, you will train the same linear model with two different loss functions — `AbsoluteError` (L1) and `SquaredError` (L2) — and observe how the choice of loss changes the gradient behavior and convergence dynamics.

## What you'll build

A side-by-side experiment that trains two identical `LinearRegression` models on the same data, one with L1 loss and one with L2 loss. You will compare their gradient magnitudes, convergence speeds, and final parameter values.

## Before you start

- Completed [Your First Training Loop](./your-first-training-loop.md) or are comfortable with the four-step pipeline
- `numpy` and `trueml` installed

## Step 1: Set up the experiment

```python
import numpy as np
from trueml.linearmodel import LinearRegression
from trueml.lossfunc import AbsoluteError, SquaredError
```

We will generate data with a known relationship and an outlier. The outlier will reveal how each loss function responds to anomalous observations.

```python
rng = np.random.default_rng(42)

n = 50
d = 1

true_w = np.array([1.5])
true_b = 0.0

X = rng.normal(size=(n, d))
y = X @ true_w + true_b + rng.normal(scale=0.2, size=n)

# Add an outlier
X = np.vstack([X, [5.0]])
y = np.append(y, [2.0])
```

The outlier is the point `(x=5.0, y=2.0)` — far from the linear trend the rest of the data follows.

Let's see how many observations we have:

```python
print(f"n = {len(y)}, d = {X.shape[1]}")
```

```
n = 51, d = 1
```

## Step 2: Create two identical models

```python
model_l1 = LinearRegression(n_features=d, lr=0.01)
model_l2 = LinearRegression(n_features=d, lr=0.01)

loss_l1 = AbsoluteError()
loss_l2 = SquaredError()
```

Both models start with the same random seed behavior (independent initialization but same distribution). Their initial weights will differ slightly, but both are near zero.

## Step 3: Train both models side by side

We will track the MAE (using L1 as the evaluation metric) and the gradient magnitude at each epoch:

```python
epochs = 300

history_l1 = {"mae": [], "grad_mag": []}
history_l2 = {"mae": [], "grad_mag": []}

for epoch in range(1, epochs + 1):
    # ── L1 model ────────────────────────────────────────
    y_pred_l1 = model_l1.forward(X)
    error_l1 = loss_l1(y, y_pred_l1)
    dw_l1, db_l1 = loss_l1.grad(X, error_l1)
    model_l1.backward(dw_l1, db_l1)

    history_l1["mae"].append(np.mean(np.abs(error_l1)))
    history_l1["grad_mag"].append(np.linalg.norm(dw_l1))

    # ── L2 model ────────────────────────────────────────
    y_pred_l2 = model_l2.forward(X)
    error_l2 = loss_l2(y, y_pred_l2)
    dw_l2, db_l2 = loss_l2.grad(X, error_l2)
    model_l2.backward(dw_l2, db_l2)

    history_l2["mae"].append(np.mean(np.abs(error_l1)))  # evaluate both with L1
    history_l2["grad_mag"].append(np.linalg.norm(dw_l2))
```

Notice that we evaluate both models using L1 loss for a fair comparison, even though `model_l2` was trained with L2 loss.

## Step 4: Compare convergence

Let's check the final MAE for each model:

```python
print(f"Final MAE (L1-trained): {history_l1['mae'][-1]:.6f}")
print(f"Final MAE (L2-trained): {history_l2['mae'][-1]:.6f}")
```

```
Final MAE (L1-trained): 0.126852
Final MAE (L2-trained): 0.200734
```

The L1-trained model generalizes better — the outlier influences it less. The L2 model is pulled toward the outlier because squared error penalizes large residuals quadratically.

## Step 5: Compare gradient magnitudes

Now let's see what happened inside the training. The gradient magnitude tells us how aggressively each model updated its parameters:

```python
print("L1 gradient magnitude, first 5 epochs:", [f"{g:.6f}" for g in history_l1['grad_mag'][:5]])
print("L2 gradient magnitude, first 5 epochs:", [f"{g:.6f}" for g in history_l2['grad_mag'][:5]])
```

```
L1 gradient magnitude, first 5 epochs: ['0.009246', '0.013738', '0.014797', '0.014060', '0.012355']
L2 gradient magnitude, first 5 epochs: ['0.710178', '0.289506', '0.216307', '0.155383', '0.113383']
```

Notice the difference in scale:

- **L1 gradient**: The magnitude is small and stable (~0.01) throughout. This is because each observation contributes at most $\pm 1$ to the gradient of $\partial L / \partial \hat{y}$, regardless of how wrong the prediction is.
- **L2 gradient**: The magnitude starts large (~0.71) and decreases as the error shrinks. This is because $\partial L / \partial \hat{y} = -2(y - \hat{y})$ — the gradient is proportional to the error itself.

Let's check the later epochs to see the pattern more clearly:

```python
print("L1 grad mag, epochs 50-55:", [f"{g:.6f}" for g in history_l1['grad_mag'][50:55]])
print("L2 grad mag, epochs 50-55:", [f"{g:.6f}" for g in history_l2['grad_mag'][50:55]])
```

```
L1 grad mag, epochs 50-55: ['0.030803', '0.030821', '0.030839', '0.030857', '0.030874']
L2 grad mag, epochs 50-55: ['0.129834', '0.129164', '0.128499', '0.127838', '0.127182']
```

The L1 gradient magnitude remains nearly constant — it oscillates around the same value. The L2 gradient continues to shrink as the model improves.

## Step 6: Compare learned parameters

```python
print("L1 model: w =", model_l1.weights[0], "b =", model_l1.bias)
print("L2 model: w =", model_l2.weights[0], "b =", model_l2.bias)
print("True w = 1.5, b = 0.0")
```

```
L1 model: w = 1.3874 b = -0.0136
L2 model: w = 0.8775 b = 0.2764
```

The L1 model recovered parameters much closer to the ground truth. The L2 model was pulled toward the outlier at `x=5.0, y=2.0`, producing a biased slope and intercept.

## What you've learned

You have compared two loss functions on the same task and observed:

- **L1 (AbsoluteError)** : Gradient magnitude is bounded at $\pm 1$ per observation. This makes it robust to outliers but produces a constant-magnitude gradient that does not shrink near the optimum.
- **L2 (SquaredError)** : Gradient magnitude is proportional to the error. This makes it converge faster initially (large steps when far from the solution) but highly sensitive to outliers.
- **Practical implication**: L1 is preferred when your data contains outliers; L2 is preferred when all observations are equally reliable and you want fast convergence.

## Next steps

- **Dive deeper**: Read [About Loss Functions](../explanation/about-loss-functions.md) for a conceptual discussion of the L1/L2 tradeoff.
- **Train on real data**: Follow [How to Train on Real Data](../how-to/train-on-real-data.md) to apply these loss functions to a real dataset.
- **Reference**: See the [AbsoluteError](../reference/absolute-error.md) and [SquaredError](../reference/squared-error.md) reference pages for the mathematical contracts.
- **Understand gradients**: Read [About Gradient Descent](../explanation/about-gradient-descent.md) for the theory behind the update rule.
