import numpy as np

from trueml.linear_model import LinearRegression
from trueml.losses import MSELoss

np.random.seed(42)

# synthetc data
x = np.random.random((1000, 5))

w = np.array([1.6, 4.1, -6.3, -1.9, 5.0]).reshape(-1, 1)
b = 0.4

y = x @ w + b


def test_lin_reg():
    model = LinearRegression(lr=0.01)
    loss_fn = MSELoss()

    initial_loss = loss_fn(y, model.forward(x))
    for _epoch in range(10000):
        y_pred = model.forward(x)
        dL_dy_pred = loss_fn.grad(y, y_pred)
        model.backward(dL_dy_pred)
    final_loss = loss_fn(y, model.forward(x))

    assert final_loss < initial_loss
    np.testing.assert_allclose(model.weights, w, atol=1e-3)
    np.testing.assert_allclose(model.bias, b, atol=1e-3)
