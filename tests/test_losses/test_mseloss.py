import numpy as np

from trueml.losses import MSELoss

y_true = np.array([[1.0], [3.0], [7.0]])
y_pred = np.array([[0.9], [2.5], [8.0]])

def test_mse_is_nonnegative(): assert MSELoss()(y_true, y_pred) >= 0
def test_mse_forward(): np.testing.assert_allclose(MSELoss()(y_true, y_pred), 0.42, rtol=1e-4)
def test_mse_grad():
  expected_grad = np.array([
    [-2 / 3 * 0.1],
    [-2 / 3 * 0.5],
    [ 2 / 3 * 1.0],
  ])
  np.testing.assert_allclose(MSELoss.grad(y_true, y_pred), expected_grad)