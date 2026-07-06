# Tutorial: Comparing Loss Functions

In the previous tutorial, you trained a model using Mean Squared Error (`MSEloss`). In this tutorial, we will explore how changing the loss function fundamentally alters the behavior of the training process, specifically in the presence of outliers.

We will compare `MSEloss` (L2 loss) and `MAEloss` (L1 loss).

## What You Will Learn
- How to swap loss functions in a TrueML training loop.
- The concept of gradient magnitude scaling.
- Why MSE is sensitive to outliers and MAE is robust to them.

---

## The Outlier Problem

Let's generate a synthetic dataset representing house sizes (features) versus house prices (targets). To make things interesting, we will inject a massive outlier: a data entry error where a small house is priced at an astronomically high value.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Normal data: 50 houses
X = np.random.uniform(1, 5, (50, 1))
# True price: $100k + $50k per unit size
y = 1.0 + 0.5 * X + np.random.randn(50, 1) * 0.1

# Inject a massive outlier (e.g. data entry error)
X = np.append(X, [[2.0]], axis=0)
y = np.append(y, [[10.0]], axis=0) # Should be ~2.0, but recorded as 10.0

plt.scatter(X[:-1], y[:-1], label="Normal Data")
plt.scatter(X[-1], y[-1], color='red', label="Outlier")
plt.legend()
plt.show()
```

## Training with MSE (L2 Loss)

First, let's train a model using `MSEloss`. Because MSE squares the errors, an error of `8.0` (from our outlier) becomes a squared penalty of `64.0`. The gradient scales linearly with the error, meaning the outlier will exert a massive pull on the model's parameters.

```python
from trueml.linearmodel import LinearRegression
from trueml.losses import MSEloss

model_mse = LinearRegression(n_features=1, lr=0.05)
mse = MSEloss()

for epoch in range(100):
    y_pred = model_mse.forward(X)
    
    # The gradient is (2/n) * (y_pred - y)
    # The outlier will have a massive gradient value
    dloss = mse.grad(y, y_pred)
    
    dw, db = model_mse.grad(X, dloss)
    model_mse.backward(dw, db)

print(f"MSE Model Weights: {model_mse.weights[0]:.3f}")
print(f"MSE Model Bias:    {model_mse.bias:.3f}")
```

If you plot this line, you will see it gets pulled significantly upwards by the red dot, ruining the fit for the rest of the normal data.

## Training with MAE (L1 Loss)

Now, let's use Mean Absolute Error (`MAEloss`). The absolute error of the outlier is `8.0`, but the *gradient* of MAE is the `sign()` of the error, which is just `1.0` or `-1.0`. The outlier exerts exactly the same amount of pull on the model as a data point that is only off by `0.1`.

```python
from trueml.losses import MAEloss

# We often use a slightly different learning rate for MAE 
# since the gradient magnitude is constant (+1 or -1)
model_mae = LinearRegression(n_features=1, lr=0.05)
mae = MAEloss()

for epoch in range(300):
    y_pred = model_mae.forward(X)
    
    # The gradient is sign(y_pred - y)
    # The outlier's gradient is just +1 or -1, same as everything else!
    dloss = mae.grad(y, y_pred)
    
    dw, db = model_mae.grad(X, dloss)
    model_mae.backward(dw, db)

print(f"MAE Model Weights: {model_mae.weights[0]:.3f}")
print(f"MAE Model Bias:    {model_mae.bias:.3f}")
```

The MAE model effectively ignores the magnitude of the outlier and fits the bulk of the data accurately.

!!! tip "Visualizing the Difference"
    TrueML provides `plot_metrics` to compare these loops visually. Try tracking the weights over time for both models. The MSE weights will jump dramatically in the first few epochs due to the outlier, while the MAE weights will walk steadily toward the true median.

## Why use MSE at all?

If MAE is so robust, why is MSE the default in machine learning?
Because MAE's constant gradient magnitude means the model never slows down as it approaches the target; it bounces around the minimum (oscillation). MSE's gradient naturally shrinks as the error shrinks, allowing the model to smoothly settle into the exact minimum.

## Next Steps

- Check the [MAEloss Reference](../reference/mae-loss.md) and [MSEloss Reference](../reference/mse-loss.md) to see the exact mathematical implementation of these gradients.
- Read the deep dive [About Loss Functions](../explanation/about-loss-functions.md) for the statistical theory (Gaussian vs Laplacian distributions) behind this behavior.
