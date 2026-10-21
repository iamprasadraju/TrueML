import numpy as np

from trueml.linear_model import LinearModel
from trueml.losses import MSELoss

np.random.seed(42)

# synthetc data
X = np.random.random((1000, 5))

w = np.array([1.6, 4.1, -6.3, -1.9, 5.0]).reshape(-1, 1)
b = 0.4

y = X @ w + b


def test_linearmodel():
  model = LinearModel(X.shape[1], y.shape[1], lr=0.01)
  loss_fn = MSELoss()

  initial_loss = loss_fn(y, model.forward(X))
  for _ in range(10000):
    y_pred = model.forward(X)
    dl_dy_pred = loss_fn.grad(y, y_pred)
    model.backward(dl_dy_pred)
  final_loss = loss_fn(y, model.forward(X))

  assert final_loss < initial_loss
  np.testing.assert_allclose(model.weights, w, atol=1e-3)
  np.testing.assert_allclose(model.bias, b, atol=1e-3)
