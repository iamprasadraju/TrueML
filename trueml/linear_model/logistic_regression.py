from __future__ import annotations

from trueml.activations import sigmoid
from trueml.linear_model.linear_model import LinearModel


class LogisticRegression(LinearModel):
    def forward(self, X):
        z = super().forward(X)
        return sigmoid(z)
