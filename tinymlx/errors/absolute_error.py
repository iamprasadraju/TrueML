def absolute_error(y_true, y_pred):
    """
    Compute the absolute error between actual and predicted values.

    Formula:
        |y_true - y_pred|

    Absolute error measures the magnitude of prediction error without
    considering its direction.

    Examples
    --------
    >>> absolute_error([10, 20], [8, 25])
    [2, 5]
    """
    return abs(y_true - y_pred)
