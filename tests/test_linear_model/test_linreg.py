import numpy as np

from trueml.linear_model import LinearRegression
from trueml.losses import MSELoss

np.random.seed(42)

# synthetc data
X = np.random.random((1000, 5))

w = np.array([1.6, 4.1, -6.3, -1.9, 5.0]).reshape(-1, 1)
b = 0.4

y = X @ w + b


def test_lin_reg():
  model = LinearRegression(X, y, lr=0.01)
  loss_fn = MSELoss()

  initial_ypred = model.forward()
  initial_loss = loss_fn(y, initial_ypred)

  model.train(loss_fn=loss_fn, epochs=10000)

  final_ypred = model.forward()
  final_loss = loss_fn(y, final_ypred)

  assert final_loss < initial_loss

  np.testing.assert_allclose(model.weights, w, atol=1e-3)
  np.testing.assert_allclose(model.bias, b, atol=1e-3)
