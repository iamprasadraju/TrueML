import numpy as np

from .linear_model import LinearModel
from ..activations import sigmoid


class LogisticRegression(LinearModel):
    """Logistic Regression model trained using explicit gradient descent.

    The model predicts probabilities using:

        p = sigmoid(Xw + b)

    where:
        X is the input feature matrix,
        w is the weight vector,
        b is the bias term.

    TrueML exposes the forward pass, gradient computation, and parameter
    updates as separate, stateless operations to make the learning pipeline
    entirely transparent.
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted probabilities for the given input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted probabilities for the positive class.
        """
        z = super().forward(X)
        return sigmoid(z)
