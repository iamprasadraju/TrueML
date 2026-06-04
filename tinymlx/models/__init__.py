"""
Models — stateless parameter containers with explicit forward/backward.

Each model holds learnable parameters (weights, bias) but never caches
training data. You must supply data at every call. This statelessness
makes dataflow explicit and prevents accidental leakage between batches.

The consistent forward/backward interface across all models lets
training loops treat any model uniformly while keeping every
mathematical operation visible.

Usage pattern:
    y_pred = model.forward(X)
    model.backward(dw, db)       # applies w -= lr * dw, b -= lr * db

Available models:
    LinearRegression       ŷ = Xw + b
    LogisticRegression     p = sigmoid(Xw + b)
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression

__all__ = [
    "LinearRegression",
    "LogisticRegression",
]
