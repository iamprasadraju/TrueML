def residual_error(y_true, y_pred):
    """
    Compute the residual error between actual and predicted values.

    Formula:
        y_true - y_pred

    Positive residuals indicate underprediction.
    Negative residuals indicate overprediction.

    Examples
    --------
    >>> residual_error([10, 20], [8, 25])
    [2, -5]
    """
    return y_true - y_pred
