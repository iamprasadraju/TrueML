import numpy as np
from numpy.typing import ArrayLike

from ..history import History


class LinearModel:
    """Linear model base class.

    This class provides the core functionality for linear models, including
    parameter initialization, forward pass, gradient computation, and parameter
    updates. It is designed to be subclassed by specific linear models (e.g.,
    :class:`~trueml.linear_model.LinearRegression`,
    :class:`~trueml.linear_model.LogisticRegression`).

    Parameters
    ----------
    n_features : int
        Number of input features (dimensionality of each sample).
    lr : float, default=0.01
        Learning rate :math:`\\eta` used during parameter updates.
    history : bool, default=True
        Whether to record training history in a :class:`~trueml.history.History` object.

    Attributes
    ----------
    weights : numpy.ndarray of shape (n_features,)
        Weight vector, initialized from :math:`\\mathcal{N}(0, 0.01^2)`.
    bias : float
        Bias (intercept) term, initialized to ``0.0``.
    lr : float
        Learning rate.
    history : History or None
        Training history recorder, or ``None`` if disabled.

    Examples
    --------
    >>> from trueml.linear_model import LinearRegression
    >>> model = LinearRegression(n_features=2)
    >>> y_pred = model.forward(X)
    >>> loss = loss_fn(y, y_pred)
    >>> dloss = loss_fn.grad(y, y_pred)
    >>> dw, db = model.grad(X, dloss)
    >>> model.backward(dw, db)
    """

    def __init__(self, n_features: int, lr: float = 0.01, history: bool = True) -> None:
        self.lr = lr
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.history = History() if history else None

    def forward(self, X: ArrayLike) -> ArrayLike:
        """Compute predictions for the given input data.

        .. math::

            \\hat{y} = X w + b

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ArrayLike of shape (n_samples,)
            Predicted values.

        Notes
        -----
        This is a pure function with respect to model parameters —
        no internal state is modified.
        """
        return X @ self.weights + self.bias

    def grad(self, X: ArrayLike, loss_gradient: ArrayLike) -> tuple[ArrayLike, float]:
        """Compute gradients of model parameters using the upstream loss gradient.

        Applies the chain rule to propagate the loss gradient back through
        the linear transformation:

        .. math::

            dw = X^\\top \\frac{\\partial L}{\\partial \\hat{y}}, \\quad
            db = \\sum_i \\frac{\\partial L_i}{\\partial \\hat{y}_i}

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input feature matrix used during the forward pass.
        loss_gradient : ArrayLike of shape (n_samples,)
            Gradient of the loss with respect to predictions
            (``dL/dy_pred``).

        Returns
        -------
        tuple[ArrayLike, float]
            A tuple ``(dw, db)`` where:

            - ``dw`` is the gradient w.r.t. the weights, shape ``(n_features,)``.
            - ``db`` is the gradient w.r.t. the bias (scalar).

        Notes
        -----
        This method only computes gradients — it does **not** update
        model parameters. Call :meth:`backward` to apply the update.
        """

        dw = X.T @ loss_gradient
        db = np.sum(loss_gradient)

        return dw, db

    def backward(self, dw: ArrayLike, db: float) -> None:
        """Update model parameters using gradient descent.

        .. math::

            w \\gets w - \\eta \\cdot dw, \\quad
            b \\gets b - \\eta \\cdot db

        Parameters
        ----------
        dw : ArrayLike of shape (n_features,)
            Gradient of the loss with respect to the weights.
        db : float
            Gradient of the loss with respect to the bias.
        """
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
