import numpy as np

from trueml.losses import MAELoss

y_true = np.array([[1.0], [3.0], [7.0]])
y_pred = np.array([[0.9], [2.5], [8.0]])

def test_mae_is_nonnegative(): assert MAELoss()(y_true, y_pred) >= 0
def test_maeloss_forward(): np.testing.assert_almost_equal(MAELoss()(y_true, y_pred), 0.5333333333333333)
def test_mae_grad(): 
  expected_grad = np.array([
    [-1 / 3], 
    [-1 / 3],
    [ 1 / 3],
  ]) 
  np.testing.assert_allclose(MAELoss.grad(y_true, y_pred), expected_grad)