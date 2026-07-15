# Metrics

**Module:** `trueml.metrics`

Evaluation scores for trained models. Pure functions that compare predictions to targets and return a single scalar score.

---

## Design Philosophy: Metrics vs. Losses

In TrueML, **losses** are differentiable objectives used during training (they define gradients). **Metrics** are for post-training evaluation and reporting — they may or may not be differentiable, and they never participate in gradient computation.

This separation prevents confusion between:
- Differentiable loss functions (used in training loops)
- Non-differentiable or differently-scaled metrics (used in evaluation)

---

## Planned Metrics

| Metric | Signature | Description |
|--------|-----------|-------------|
| `mae` | `mae(y, y_pred) → float` | Mean Absolute Error |
| `mse` | `mse(y, y_pred) → float` | Mean Squared Error |
| `accuracy` | `accuracy(y, y_pred) → float` | Classification accuracy |

!!! note "Under Development"
    The metrics module is currently being implemented. The function signatures above represent the planned API.

---

## API Reference

::: trueml.metrics
    options:
      show_source: true
      heading_level: 3

## See Also
- [MSEloss](losses/mean_squared_error.md) — Differentiable MSE loss for training.
- [MAEloss](losses/mean_absolute_error.md) — Differentiable MAE loss for training.
