import numpy as np


class LinearRegression:
    """
    Linear Regression model trained using gradient descent.

    The model predicts targets using:

        y = Xw + b

    where:
        X is the input feature matrix,
        w is the weight vector,
        b is the bias term.

    TinyMLx exposes the forward pass, gradient computation,
    and parameter updates as separate operations to make the
    learning process explicit.
    """

    def __init__(self, n_features, lr=0.01):
        """
        Initialize model parameters.

        Parameters
        ----------
        n_features : int
            Number of input features.

        lr : float, default=0.01
            Learning rate used during parameter updates.
        """
        self.lr = lr
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0.0

    def forward(self, X):
        """
        Compute predictions for the given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        array-like
            Predicted values.

        Notes
        -----
        The input data is stored internally and reused during
        gradient computation.
        """
        self.X = X
        return X @ self.weights + self.bias

    def grad(self, loss_gradient):
        """
        Compute gradients of model parameters using the upstream
        loss gradient.

        Parameters
        ----------
        loss_gradient : array-like of shape (n_samples,)
            Gradient of the loss with respect to predictions
            (dL/dy_pred).

        Returns
        -------
        tuple
            A tuple (dw, db) where:

            - dw is the gradient with respect to the weights.
            - db is the gradient with respect to the bias.

        Notes
        -----
        This method assumes that ``forward()`` has already been
        called. The input data from the forward pass is reused to
        compute parameter gradients.

        Examples
        --------
        >>> y_pred = model.forward(X)
        >>> dloss = mse.grad(y_true, y_pred)
        >>> dw, db = model.grad(dloss)
        """
        if not hasattr(self, "X"):
            raise RuntimeError(
                "grad() called before forward(). "
                "Call forward(X) first."
            )

        n = self.X.shape[0]

        dw = (1 / n) * (self.X.T @ loss_gradient)
        db = (1 / n) * np.sum(loss_gradient)

        return dw, db

    def backward(self, dw, db):
        """
        Update model parameters using gradient descent.

        Parameters
        ----------
        dw : array-like
            Gradient of the loss with respect to the weights.

        db : float
            Gradient of the loss with respect to the bias.

        Notes
        -----
        Parameters are updated according to:

            w = w - lr * dw
            b = b - lr * db
        """
        self.weights -= self.lr * dw
        self.bias -= self.lr * db