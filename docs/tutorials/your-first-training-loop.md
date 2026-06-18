# Your First Training Loop

In this tutorial, you will train a `LinearRegression` model from scratch using gradient descent. By the end, you will have written every line of a working training loop and seen the model converge from random parameters to a useful predictor.

## What you'll build

A complete training loop that learns the mapping $y = 2x_1 - 3x_2 + 0.5$ from synthetic data, using `AbsoluteError` loss and manual gradient descent.

The final script will be about 20 lines of Python — and every one of them will be under your control.

## Before you start

- Python 3.10+ with `numpy` and `tinymlx` installed
- Familiarity with basic NumPy array operations (shapes, indexing, `@` operator)

## Step 1: Import the library

```python
import numpy as np
from tinymlx.linearmodel import LinearRegression
from tinymlx.lossfunc import AbsoluteError
```

Run this cell. If you see no errors, you're ready.

## Step 2: Create synthetic data

We need data with a known relationship so we can check whether the model learns the right parameters.

```python
rng = np.random.default_rng(42)

n = 100          # number of observations
d = 2            # number of features

true_w = np.array([2.0, -3.0])
true_b = 0.5

X = rng.normal(size=(n, d))
y = X @ true_w + true_b + rng.normal(scale=0.1, size=n)
```

Your `X` should have shape `(100, 2)` and `y` should have shape `(100,)`. Let's verify:

```python
print("X shape:", X.shape)
print("y shape:", y.shape)
```

The output should look like:

```
X shape: (100, 2)
y shape: (100,)
```

Notice that `y` is a noisy version of `X @ true_w + true_b`. The noise (standard deviation 0.1) simulates measurement error, so the model cannot recover the exact ground truth — only approximate it.

## Step 3: Initialize the model and loss function

```python
model = LinearRegression(n_features=d, lr=0.01)
loss_fn = AbsoluteError()
```

The model's weights are initialized randomly (small normal values) and the bias starts at zero. Let's inspect the initial parameters:

```python
print("Initial weights:", model.weights)
print("Initial bias:  ", model.bias)
```

The output should look something like:

```
Initial weights: [ 0.0043 -0.0089]
Initial bias:    0.0
```

Your exact values will differ (random seed), but they should be near zero.

## Step 4: Write the forward pass

The first step inside the training loop is the forward pass: compute predictions.

```python
y_pred = model.forward(X)
```

`y_pred` now holds $\hat{y} = Xw + b$ for every observation. Let's see what the untrained model predicts:

```python
print("First 5 predictions:", y_pred[:5])
print("First 5 true values:", y[:5])
```

The output should look something like:

```
First 5 predictions: [ 0.0152 -0.0089 -0.0162  0.0312 -0.0012]
First 5 true values: [ 3.0614 -6.7174  0.3490  7.1130 -7.3924]
```

The predictions are near zero (because weights are near zero) while the true values span a wide range. The model is not yet useful.

## Step 5: Compute the loss

```python
error = loss_fn(y, y_pred)
```

`error` contains the element-wise absolute error $|y_i - \hat{y}_i|$ for each observation. The mean absolute error (MAE) gives a single-number summary:

```python
print("MAE:", np.mean(error))
```

```
MAE: 3.0157
```

A MAE of ~3 means the model is off by about 3 units on average — poor performance, as expected from an untrained model.

## Step 6: Compute gradients

```python
dw, db = loss_fn.grad(X, error)
```

This single call computes the partial derivatives of the loss with respect to both the weights and the bias, via the chain rule. Let's inspect the gradient:

```python
print("dw:", dw)
print("db:", db)
```

```
dw: [-0.01413086  0.00854147]
db: -0.014
```

These gradients tell us which direction increases the loss. To *reduce* the loss, we need to move the parameters in the opposite direction.

## Step 7: Update the parameters

```python
model.backward(dw, db)
```

This subtracts `lr * dw` from the weights and `lr * db` from the bias. After one update, let's check the new parameters:

```python
print("Updated weights:", model.weights)
print("Updated bias:  ", model.bias)
```

```
Updated weights: [ 0.0144 -0.0175]
Updated bias:    0.00014
```

The weights moved away from zero in the first step. They should be moving toward the true values `[2.0, -3.0]`.

## Step 8: Assemble the full training loop

Repeat steps 4-7 for many epochs:

```python
epochs = 500
report_interval = 50

for epoch in range(1, epochs + 1):
    y_pred = model.forward(X)
    error = loss_fn(y, y_pred)
    dw, db = loss_fn.grad(X, error)
    model.backward(dw, db)

    if epoch % report_interval == 0:
        mae = np.mean(np.abs(error))
        print(f"epoch {epoch:4d}  MAE = {mae:.6f}")
```

Run the loop. The output should show the MAE decreasing:

```
epoch   50  MAE = 0.493714
epoch  100  MAE = 0.317615
epoch  150  MAE = 0.244057
epoch  200  MAE = 0.200163
epoch  250  MAE = 0.170503
epoch  300  MAE = 0.149089
epoch  350  MAE = 0.133041
epoch  400  MAE = 0.120960
epoch  450  MAE = 0.111730
epoch  500  MAE = 0.104453
```

The MAE drops from ~3.0 to ~0.1 — the model is learning. The improvements are large at first and gradually taper off.

## Step 9: Inspect the learned parameters

After training, compare the learned parameters to the ground truth:

```python
print("Learned weights:", model.weights)
print("True weights:   ", true_w)
print()
print("Learned bias:", model.bias)
print("True bias:   ", true_b)
```

```
Learned weights: [ 1.9985 -2.9901]
True weights:    [ 2.0 -3.0]

Learned bias: 0.4964
True bias:    0.5
```

The model has approximately recovered the data-generating parameters. The small discrepancies come from the noise we added to `y` and the fact that gradient descent only approximates the optimal solution in finite steps.

## What you've learned

You have written a complete training loop: forward pass, loss computation, gradient computation, and parameter update. You have seen:

- How random initial parameters produce poor predictions
- How gradients point in the direction of increasing loss
- How gradient descent moves parameters to reduce the loss
- How repeated updates cause the model to converge toward the true parameters

This four-step sequence — forward, loss, gradient, backward — is the fundamental pattern of supervised learning in TinyMLx. Every experiment you build will follow this same structure.

## Next steps

- **Debug gradient issues**: See the [Calculus Mapping](../explanation/calculus-mapping.md) explanation for common gradient pitfalls.
- **Train on real data**: Follow the [How to Train on Real Data](../how-to/train-on-real-data.md) guide.
- **Understand the math**: Read [About Gradient Descent](../explanation/about-gradient-descent.md) for the theory behind this procedure.
- **Switch loss functions**: Try the [SquaredError](../reference/squared-error.md) reference to see how L2 loss changes convergence behavior.
