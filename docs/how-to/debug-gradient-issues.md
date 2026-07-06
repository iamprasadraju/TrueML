# How-to: Debug Gradient Issues

Because TrueML exposes every numerical operation as a NumPy array, it is the perfect environment to learn how to diagnose and fix gradient descent failures. If your model's loss becomes `NaN` or refuses to decrease, the problem is almost always in the gradients.

This guide provides a diagnostic framework and practical solutions for common optimization failures.

## Symptom Table

| Symptom | Diagnosis | Likely Culprits |
|---------|-----------|-----------------|
| Loss immediately goes to `NaN` | Exploding Gradient | Learning rate too high, unscaled features. |
| Loss drops initially, then violently bounces up and down | Oscillating Gradient | Learning rate slightly too high, or L1 loss behavior. |
| Loss barely changes, weights don't update | Vanishing Gradient | Learning rate too low. |
| Loss steadily increases | Divergence | Gradient points wrong way (custom loss bug) or LR too high. |

---

## Issue 1: Exploding Gradients

**The problem:** The parameter update ($dw$) is so massive that the new weights exceed floating-point maximums, resulting in `NaN` (Not a Number) or `Inf`.

**How to diagnose in TrueML:**
Because TrueML exposes the gradient directly, you can simply print it. If you see massive numbers ($10^6$ or higher), your gradients are exploding.

```python
# Inside your training loop:
dw, db = model.grad(X, dloss)
print("Max dw:", np.max(np.abs(dw))) # If this is massive, you have a problem.
```

**Solutions:**

1. **Standardize your features.** If a feature has values in the millions (like House Price), the gradient for that weight will be multiplied by millions. (See [Train on Real Data](train-on-real-data.md)).
2. **Lower the learning rate.** Drop it by a factor of 10 (`lr=0.01` to `lr=0.001`) and observe.
3. **Manual Gradient Clipping.** Because TrueML returns `dw` to you before updating, you can clip the array yourself!

```python
# Solution 3: Manual Gradient Clipping
dloss = loss_fn.grad(y, y_pred)
dw, db = model.grad(X, dloss)

# Clip the gradients to a maximum magnitude of 5.0
clip_val = 5.0
dw = np.clip(dw, -clip_val, clip_val)
db = np.clip(db, -clip_val, clip_val)

model.backward(dw, db)
```

---

## Issue 2: Oscillating Gradients

**The problem:** The model repeatedly overshoots the minimum. The loss drops, then jumps up, then drops, then jumps up.

**How to diagnose in TrueML:**
Plot the loss curve using `plot_metrics`. If it looks like a zigzag pattern, the model is oscillating.

**Solutions:**

1. **Lower the learning rate.** Oscillation means the step size is too large to settle into the valley.
2. **Check your loss function.** If you are using `MAEloss` (L1), oscillation near the minimum is a mathematical guarantee because the gradient magnitude never shrinks (it is always $\pm 1$). Switch to `MSEloss` if you need perfect convergence, or decay your learning rate over time.

```python
# Solution 2b: Manual Learning Rate Decay
for epoch in range(1000):
    # Decay the learning rate linearly over time
    current_lr = 0.1 * (1 - (epoch / 1000))
    model.lr = current_lr # TrueML lets you mutate this freely!
    
    # ... rest of training loop ...
```

---

## Issue 3: Vanishing Gradients

**The problem:** The gradients are so close to zero that the weights barely change.

**How to diagnose in TrueML:**
Print the gradients. If they are in the range of $10^{-6}$, your model is barely learning.

```python
# Inside your training loop:
dw, db = model.grad(X, dloss)
print("Mean dw magnitude:", np.mean(np.abs(dw))) 
```

**Solutions:**

1. **Increase the learning rate.** 
2. **Check the loss.** If your model is already making near-perfect predictions, the gradient naturally vanishes (this is a good thing!).
3. **Check for dead activations.** If you are using `trueml.activations.sigmoid` for a regression task where targets are outside the $(0, 1)$ range, the sigmoid will saturate (output exactly 0 or 1), driving its local derivative to 0, killing the gradient. Ensure you match the activation to the domain of your targets.

## Summary

TrueML's "No Abstraction" philosophy makes debugging gradients trivial. You do not need complex hooking mechanisms to inspect gradients; they are returned to you as standard NumPy arrays in the middle of your `for` loop. If something is broken, `print(dw)` is your best friend.
