# How to Train on a Real Dataset

This guide shows you how to load the Housing.csv dataset, prepare it for a linear model, and train a `LinearRegression` model using gradient descent.

## When to use this guide

Use this when you have a real-world tabular dataset and want to train a linear model with TrueML. This guide covers data loading, feature selection, normalization, and the training loop.

## Before you start

- Completed [Your First Training Loop](../tutorials/your-first-training-loop.md) or are familiar with the four-step pipeline
- The `datasets/Housing.csv` file exists in your TrueML installation

## Context

Real data differs from synthetic data in important ways:
- Features have different scales (area ranges in thousands, bedrooms in single digits)
- Data may contain outliers
- The relationship between features and target is unknown

Proper preprocessing — especially feature standardization — is essential for gradient descent to converge efficiently.

## Steps

### 1. Load and inspect the data

```python
import numpy as np
import pandas as pd
from trueml.linearmodel import LinearRegression
from trueml.lossfunc import AbsoluteError

data = pd.read_csv("datasets/Housing.csv")
print(data.head())
print(data.dtypes)
```

The dataset has 545 rows with a mix of numerical and categorical columns. For this guide we will use four numerical features:

```
price         int64
area          int64
bedrooms      int64
bathrooms     int64
stories       int64
mainroad     object
...
```

### 2. Select features and target

```python
feature_cols = ["area", "bedrooms", "bathrooms", "stories"]
target_col = "price"

X = data[feature_cols].values.astype(float)
y = data[target_col].values.astype(float)
```

### 3. Standardize the features

Gradient descent is sensitive to feature scale. If one feature has values in the thousands (area) and another has values in single digits (bedrooms), the gradient will be dominated by the larger-scale feature.

```python
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std
```

Standardization centers each feature at zero mean and unit variance. This ensures each feature contributes proportionally to the gradient.

### 4. Initialize the model

```python
n, d = X.shape
model = LinearRegression(n_features=d, lr=0.1)
loss_fn = AbsoluteError()
```

The learning rate is higher (0.1) than the default (0.01) because the standardized features have consistent scale, allowing larger steps without divergence.

### 5. Train the model

```python
epochs = 200
report_interval = 20

for epoch in range(1, epochs + 1):
    y_pred = model.forward(X)
    error = loss_fn(y, y_pred)
    dw, db = loss_fn.grad(X, error)
    model.backward(dw, db)

    if epoch % report_interval == 0:
        mae = np.mean(np.abs(error)) * y_std
        print(f"epoch {epoch:4d}  MAE = ₹{mae:.0f}")
```

Output (approximate):

```
epoch   20  MAE = ₹1315283
epoch   40  MAE = ₹1248652
epoch   60  MAE = ₹1225190
epoch   80  MAE = ₹1212996
epoch  100  MAE = ₹1206361
epoch  120  MAE = ₹1202915
epoch  140  MAE = ₹1201085
epoch  160  MAE = ₹1200080
epoch  180  MAE = ₹1199507
epoch  200  MAE = ₹1199145
```

The MAE is reported in the original price scale (rupees) by multiplying the normalized error by `y_std`.

### 6. Evaluate the learned parameters

```python
# Rescale weights to original units
w_rescaled = model.weights / X_std * y_std
b_rescaled = model.bias * y_std + y_mean - (X_mean / X_std * y_std) @ model.weights

print("Feature coefficients:")
for col, coef in zip(feature_cols, w_rescaled):
    print(f"  {col}: ₹{coef:.0f} per unit")
print(f"Intercept: ₹{b_rescaled:.0f}")
```

Example output:

```
Feature coefficients:
  area: ₹840 per unit
  bedrooms: ₹175000 per unit
  bathrooms: ₹550000 per unit
  stories: ₹530000 per unit
Intercept: ₹-320000
```

## Troubleshooting

**Problem: Loss diverges (MAE increases instead of decreasing)**
Solution: Your learning rate is too high. Reduce `lr` (try 0.01 or 0.001) and retrain.

**Problem: Loss converges very slowly**
Solution: Your learning rate is too low, or features are not standardized. Check that `X` has zero mean and unit variance.

## Variations

**Using SquaredError instead of AbsoluteError:**
Replace `AbsoluteError()` with `SquaredError()` and reduce the learning rate to 0.01. SquaredError produces larger gradients for large errors, so it requires a smaller step size.

**Training on all features including categorical:**
Convert categorical columns (mainroad, guestroom, etc.) to 0/1 indicators using `pd.get_dummies(data)`, then standardize and train.

## Related guides

- [How to Implement Minibatch Gradient Descent](./implement-minibatch-gd.md)
- [Manual Gradient Descent](./manual-gradient-descent.md)
- [AbsoluteError Reference](../reference/absolute-error.md)
- [About Gradient Descent](../explanation/about-gradient-descent.md)
