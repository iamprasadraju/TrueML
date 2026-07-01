"""
Linear models — supervised parameter containers.

Each model holds parameters (weights, bias) and exposes:
    forward(X)  → y_pred   — compute the prediction
    backward(dw, db) → None — apply gradient descent update

No .fit() method, no cached training data. The user writes the
training loop, keeping every mathematical operation visible.
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression

__all__ = [
    "LinearRegression",
    "LogisticRegression",
]
