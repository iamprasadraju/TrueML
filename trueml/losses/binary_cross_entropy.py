from __future__ import annotations

import numpy as np


class BCELoss:
    """Binary cross-entropy (log loss) for binary classification.

    Derived from Negative log-likelihood (NLL) under Bernoulli distribution
    via Maximum likelihood estimation (MLE):
        P(Y=y|ŷ) = ŷ^y * (1 - ŷ)^(1 - y)
        Loss = -ln P(Y|ŷ) = -[y*ln(ŷ) + (1 - y)*ln(1 - ŷ)]

    Analytical gradient with respect to predictions (ŷ):
        ∂L/∂ŷ = -[y/ŷ - (1 - y)/(1 - ŷ)]
              = -[(y(1 - ŷ) - ŷ(1 - y)) / (ŷ(1 - ŷ))]
              = (ŷ - y) / (ŷ(1 - ŷ))

    References:
        - Wikipedia (Cross-entropy & logistic regression):
          https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression
        - Deep learning (Goodfellow et al., Ch. 6.2.2.2 Sigmoid units for
          Bernoulli output distributions Pg no. 178):
          https://www.deeplearningbook.org/contents/mlp.html
    """

    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def grad(y_true, y_pred, eps: float = 1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        n = y_true.size
        dL_dy_pred = (1 / n) * (y_pred - y_true) / (y_pred * (1 - y_pred))
        return dL_dy_pred
