from __future__ import annotations

from ..activations import sigmoid
from .linear_model import LinearModel


class LogisticRegression(LinearModel):
    def forward(self, X):
        z = super().forward(X)
        return sigmoid(z)
