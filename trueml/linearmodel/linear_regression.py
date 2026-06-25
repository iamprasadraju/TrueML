import numpy as np


class LinearRegression:
    """Linear Regression model trained using explicit gradient descent.

    The model predicts targets using:
        y = Xw + b

    where:
        X is the input feature matrix,
        w is the weight vector,
        b is the bias term.

    TrueML exposes the forward pass, gradient computation, and parameter
    updates as separate, stateless operations to make the learning pipeline
    entirely transparent.
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
        # Using randn (Gaussian distribution) is a more standard ML initialization
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions for the given input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted values.
        """
        return X @ self.weights + self.bias

    def grad(self, X: np.ndarray, loss_gradient: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute gradients of model parameters using the upstream loss gradient.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix used during the forward pass.
        loss_gradient : np.ndarray of shape (n_samples,)
            Gradient of the loss with respect to predictions (dL/dy_pred).

        Returns
        -------
        tuple[np.ndarray, float]
            A tuple (dw, db) where:
            - dw is the gradient with respect to the weights (shape: n_features).
            - db is the gradient with respect to the bias (scalar).

        Examples
        --------
        >>> model = LinearRegression(n_features=2)
        >>> y_pred = model.forward(X)
        >>> dloss = 2 * (y_pred - y_true)  # Explicit MSE gradient
        >>> dw, db = model.grad(X, dloss)
        """
        n = X.shape[0]

        dw = X.T @ loss_gradient
        db = np.sum(loss_gradient)

        return dw, db

    def backward(self, dw: np.ndarray, db: float) -> None:
        """Update model parameters using gradient descent optimization.

        Parameters
        ----------
        dw : np.ndarray
            Gradient of the loss with respect to the weights.
        db : float
            Gradient of the loss with respect to the bias.
        """
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
