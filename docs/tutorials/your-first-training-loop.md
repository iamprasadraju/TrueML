# Tutorial: Your First Training Loop

Welcome to TrueML! In this tutorial, you will build and train a linear regression model from scratch. Unlike other frameworks where you call `.fit()` and wait for a result, TrueML requires you to write the training loop yourself. By the end of this page, you will understand exactly how data flows from prediction, to loss, to gradient, to parameter update.

## What You Will Learn
- How to initialize a `LinearRegression` model.
- How to compute predictions (the forward pass).
- How to evaluate those predictions using `MSEloss`.
- How the chain rule connects the loss gradient to the model's parameter gradients.
- How to update the model using gradient descent.

## Prerequisites
- Basic familiarity with Python and NumPy.
- Conceptual understanding of what linear regression does.
- TrueML installed in your environment.

---

## Step 1: Prepare the Data

Machine learning requires data. Let's create a synthetic dataset where we know the true underlying relationship. We will generate 100 samples with 1 feature, and add a little bit of random noise.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate 100 random values for our single feature X
np.random.seed(42)
X = np.random.randn(100, 1)

# True relationship: y = 2.5 * X + 1.0 (plus some noise)
y_true = 2.5 * X + 1.0 + np.random.randn(100, 1) * 0.2

plt.scatter(X, y_true, alpha=0.5)
plt.title("Synthetic Dataset")
plt.show()
```

## Step 2: Initialize the Model and Loss

We will use the `LinearRegression` model and the Mean Squared Error (`MSEloss`) function. The model requires the number of features (1 in this case) and a learning rate (how big of a step to take during updates).

```python
from trueml.linearmodel import LinearRegression
from trueml.losses import MSEloss

# Initialize model with 1 feature and learning rate 0.1
model = LinearRegression(n_features=1, lr=0.1)

# Initialize the loss function
loss_fn = MSEloss()

print("Initial weight:", model.weights)
print("Initial bias:", model.bias)
```

!!! note "Random Initialization"
    Your initial weight will be a random number very close to zero, and your initial bias is `0.0`. The model knows nothing about the data yet.

## Step 3: The Four-Step Loop

The core of TrueML is the training loop. Every iteration (epoch) over the data involves four explicit steps. Let's write the loop and trace the math.

```python
epochs = 50
loss_history = []

for epoch in range(epochs):
    # ----------------------------------------------------
    # 1. FORWARD PASS
    # The model predicts y using its current weights.
    # Math: y_pred = X * w + b
    # ----------------------------------------------------
    y_pred = model.forward(X)
    
    # ----------------------------------------------------
    # 2. LOSS COMPUTATION
    # Evaluate how wrong the predictions are.
    # Math: L = mean((y_true - y_pred)^2)
    # ----------------------------------------------------
    loss_value = loss_fn(y_true, y_pred)
    loss_history.append(loss_value)
    
    # ----------------------------------------------------
    # 3. GRADIENT COMPUTATION
    # a) How does the loss change with respect to predictions?
    loss_grad = loss_fn.grad(y_true, y_pred)
    
    # b) Chain rule: How does the loss change with respect to weights?
    dw, db = model.grad(X, loss_grad)
    
    # ----------------------------------------------------
    # 4. BACKWARD PASS
    # Update the parameters using the gradients.
    # Math: w = w - lr * dw
    # ----------------------------------------------------
    model.backward(dw, db)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d} | Loss: {loss_value:.4f}")

print("\nFinal weight:", model.weights)
print("Final bias:", model.bias)
```

**Expected Output:**
```text
Epoch 10 | Loss: 0.5843
Epoch 20 | Loss: 0.1064
Epoch 30 | Loss: 0.0463
Epoch 40 | Loss: 0.0387
Epoch 50 | Loss: 0.0378

Final weight: [2.441]
Final bias: 0.985
```

The model successfully recovered the underlying parameters (~2.5 for weight, ~1.0 for bias)!

## What You've Learned

You just trained a model without using `.fit()`. You explicitly executed the mathematical contract of machine learning:
1. Generate predictions (`forward`).
2. Measure the error (`__call__`).
3. Compute derivatives using the chain rule (`grad`).
4. Step in the direction that minimizes the error (`backward`).

Because TrueML exposes these primitive operations as NumPy arrays, you could insert `print(loss_grad)` or `print(dw)` inside the loop at any time to inspect the exact mathematical state of the optimization.

## Next Steps

- **Compare losses:** See what happens when the data contains extreme outliers in the [Comparing Loss Functions](comparing-loss-functions.md) tutorial.
- **Deepen the math:** Read the [Calculus Mapping](../explanation/calculus-mapping.md) explanation to see the rigorous derivation of the chain rule used in step 3.
- **Explore the API:** Check the [LinearRegression Reference](../reference/linear-regression.md) for the formal class contract.
