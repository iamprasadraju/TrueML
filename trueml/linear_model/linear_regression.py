from .linear_model import LinearModel


class LinearRegression(LinearModel):
    """Linear Regression model trained using explicit gradient descent.

    Predicts continuous targets using the linear equation:

    .. math::

        \\hat{y} = X w + b

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
    LogisticRegression : For binary classification tasks.
    trueml.losses.MSEloss : Mean Squared Error loss.
    trueml.losses.MAEloss : Mean Absolute Error loss.

    Notes
    -----
    Unlike scikit-learn's ``LinearRegression.fit()``, there is no
    closed-form solution (Normal Equation) computed here. This model
    strictly uses iterative gradient descent.

    Examples
    --------
    >>> import numpy as np
    >>> from trueml.linear_model import LinearRegression
    >>> from trueml.losses import MSEloss
    >>>
    >>> X = np.random.randn(100, 3)
    >>> y = X @ np.array([1.5, -2.0, 0.5]) + 0.1
    >>>
    >>> model = LinearRegression(n_features=3, lr=0.01)
    >>> loss_fn = MSEloss()
    >>>
    >>> for epoch in range(500):
    ...     y_pred = model.forward(X)
    ...     loss = loss_fn(y, y_pred)
    ...     dw, db = model.grad(X, loss_fn.grad(y, y_pred))
    ...     model.backward(dw, db)
    """

    pass
