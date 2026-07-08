import numpy as np

from .linear_model import LinearModel


class LinearRegression(LinearModel):
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

    pass
