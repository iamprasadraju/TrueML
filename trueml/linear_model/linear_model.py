import numpy as np
from numpy.typing import ArrayLike

class LinearModel:
    """Linear model base class.

    This class provides the core functionality for linear models, including
    parameter initialization, forward pass, gradient computation, and parameter
    updates. It is designed to be subclassed by specific linear models (e.g.,
    LinearRegression, LogisticRegression).


    Example
    -------
    >>> from trueml.linearmodel import LinearRegression
    >>> model = LinearRegression(n_features=2)
    >>> y_pred = model.forward(X)
    >>> loss = loss_fn(y, y_pred)
    >>> dloss = loss_fn.grad(y, y_pred)
    >>> dw, db = model.grad(X, dloss)
    >>> model.backward(dw, db)
    """

    def __init__(self, n_features: int, lr: float = 0.01) -> None:
        """Initialize model parameters.

        Parameters
        ----------
        n_features : int
            Number of input features.
        lr : float, default=0.01
            Learning rate used during parameter updates.
        """
        self.lr = lr
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    def forward(self, X: ArrayLike) -> ArrayLike:
        """Compute predictions for the given input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ArrayLike of shape (n_samples,)
            Predicted values.
        """
        return X @ self.weights + self.bias

    def grad(
        self, X: ArrayLike, loss_gradient: ArrayLike
    ) -> tuple[ArrayLike, float]:
        """Compute gradients of model parameters using the upstream loss gradient.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input feature matrix used during the forward pass.
        loss_gradient : ArrayLike of shape (n_samples,)
            Gradient of the loss with respect to predictions (dL/dy_pred).

        Returns
        -------
        tuple[ArrayLike, float]
            A tuple (dw, db) where:
            - dw is the gradient with respect to the weights (shape: n_features).
            - db is the gradient with respect to the bias (scalar).

        """

        dw = X.T @ loss_gradient
        db = np.sum(loss_gradient)

        return dw, db

    def backward(self, dw: ArrayLike, db: float) -> None:
        """Update model parameters using gradient descent optimization.

        Parameters
        ----------
        dw : ArrayLike
            Gradient of the loss with respect to the weights.
        db : float
            Gradient of the loss with respect to the bias.
        """
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
