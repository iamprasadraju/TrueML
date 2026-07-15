from numpy.typing import ArrayLike

from ..activations import sigmoid
from .linear_model import LinearModel


class LogisticRegression(LinearModel):
    """Logistic Regression model trained using explicit gradient descent.

    Predicts class probabilities using the sigmoid of the linear equation:

    .. math::

        p = \\sigma(X w + b) = \\frac{1}{1 + e^{-(X w + b)}}

    where :math:`X` is the input feature matrix, :math:`w` is the weight
    vector, and :math:`b` is the bias term.

    TrueML exposes the forward pass, gradient computation, and parameter
    updates as separate, stateless operations to make the learning
    pipeline entirely transparent.

    Parameters
    ----------
    n_features : int
        Number of input features.
    lr : float, default=0.01
        Learning rate for gradient descent.
    history : bool, default=True
        Whether to record training history.

    See Also
    --------
    LinearRegression : For continuous regression tasks.
    trueml.activations.sigmoid : The activation function used.

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.linear_model import LogisticRegression
    >>>
    >>> X = np.random.randn(100, 2)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(float)
    >>>
    >>> model = LogisticRegression(n_features=2, lr=0.1)
    >>> probs = model.forward(X)
    """

    def forward(self, X: ArrayLike) -> ArrayLike:
        """Compute predicted probabilities for the given input data.

        Applies the sigmoid activation to the linear output:

        .. math::

            p = \\sigma(X w + b)

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ArrayLike of shape (n_samples,)
            Predicted probabilities for the positive class,
            each in the range :math:`(0, 1)`.
        """
        z = super().forward(X)
        return sigmoid(z)
