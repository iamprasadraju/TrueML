# How-to: Train on Real Data

Training on synthetic data (like `np.random.randn`) is easy because the features are naturally scaled and centered. When applying TrueML to real-world datasets, features often have wildly different scales (e.g., age in years vs. income in dollars). 

This guide demonstrates how to load a real dataset, properly standardize the features, train a TrueML model, and interpret the learned parameters.

## When to use this guide
- You are transitioning from synthetic data to CSVs.
- Your model is diverging (loss going to `NaN`) and you don't know why.
- You need to rescale learned weights back to original units for interpretation.

---

## 1. Loading and Standardizing Data

Gradient descent is highly sensitive to the scale of the input features. If feature $x_1$ ranges from $0-1$ and feature $x_2$ ranges from $1000-50000$, the gradient for $x_2$ will dominate the update step, causing oscillation or divergence.

Always standardize features to have mean $= 0$ and standard deviation $= 1$.

```python
import numpy as np
import pandas as pd
from trueml.linearmodel import LinearRegression
from trueml.losses import MSEloss

# 1. Load data
# Assume housing.csv has columns: 'SquareFeet', 'Bedrooms', 'Age', and 'Price'
df = pd.read_csv("datasets/housing.csv")

X_raw = df[['SquareFeet', 'Bedrooms', 'Age']].values
y_raw = df['Price'].values

# 2. Standardize Features (Z-score normalization)
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)

X_scaled = (X_raw - X_mean) / X_std

# Standardizing targets is optional but helps with learning rate stability
y_mean = np.mean(y_raw)
y_std = np.std(y_raw)

y_scaled = (y_raw - y_mean) / y_std
```

!!! warning "Keep the Scaling Factors"
    You must save `X_mean`, `X_std`, `y_mean`, and `y_std`. You will need them to make predictions on new data, and to interpret the weights later!

---

## 2. The Training Loop

Now we apply the standard TrueML training loop on the scaled data.

```python
model = LinearRegression(n_features=3, lr=0.1)
loss_fn = MSEloss()

for epoch in range(1, 201):
    y_pred = model.forward(X_scaled)
    
    loss = loss_fn(y_scaled, y_pred)
    dloss = loss_fn.grad(y_scaled, y_pred)
    
    dw, db = model.grad(X_scaled, dloss)
    model.backward(dw, db)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Scaled MSE: {loss:.4f}")
```

---

## 3. Rescaling for Interpretation

The weights stored in `model.weights` correspond to the *scaled* data. A weight of `0.5` for 'Bedrooms' means "A 1 standard deviation increase in Bedrooms results in a 0.5 standard deviation increase in Price."

To convert the weights back to original units (e.g., "Dollars per Square Foot"), apply the inverse transformation:

$$ w_{original} = w_{scaled} \times \left( \frac{\sigma_y}{\sigma_x} \right) $$

$$ b_{original} = (b_{scaled} \times \sigma_y + \mu_y) - \sum (w_{original} \times \mu_x) $$

```python
# Rescale weights
w_scaled = model.weights
w_original = w_scaled * (y_std / X_std)

# Rescale bias
b_scaled = model.bias
b_original = (b_scaled * y_std + y_mean) - np.sum(w_original * X_mean)

print("Original Units:")
print(f"Base Price (Bias): ${b_original:,.2f}")
print(f"Per Square Foot:   ${w_original[0]:,.2f}")
print(f"Per Bedroom:       ${w_original[1]:,.2f}")
print(f"Per Year of Age:   ${w_original[2]:,.2f}")
```

## Summary
Real data requires preprocessing. Because TrueML lacks hidden `.fit()` heuristics, it will not automatically scale data or adjust learning rates for you. You must explicitly control the data statistics entering the model.
