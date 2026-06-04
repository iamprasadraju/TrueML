"""
Metrics — evaluation scores for trained models.

Pure functions that compare predictions to targets and return a
single scalar score. Not used in gradient computation (use lossfunc
for that) — these are for post-training evaluation and reporting.

The separation prevents confusion between differentiable loss
functions (used in training) and non-differentiable or differently-
scaled metrics (used in evaluation).

Available metrics:
    mae(y, y_pred)    — mean absolute error
    mse(y, y_pred)    — mean squared error
    accuracy(y, y_pred) — classification accuracy
"""
