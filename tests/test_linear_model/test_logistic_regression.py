import numpy as np

from trueml.activations import Sigmoid
from trueml.linear_model import LogisticRegression
from trueml.losses import BCELoss, MSELoss


def test_logistic_regression_with_bce():
  """
  Verify LogisticRegression trains and converges with Sigmoid and BCELoss.
  """
  np.random.seed(42)

  # Shift cluster centers to (+-1.5) to guarantee clean linear separability:
  # eliminates data ambiguity so any test failure strictly pinpoints gradient bugs.
  X0 = np.random.randn(100, 2) - 1.5
  X1 = np.random.randn(100, 2) + 1.5
  X = np.vstack([X0, X1])
  y = np.vstack([np.zeros((100, 1)), np.ones((100, 1))])

  # lr=0.1 is sufficient for BCE because the activation gradient
  # y_pred * (1 - y_pred) cancels with the BCE denominator,
  # keeping the effective gradient linear (y_pred - y).
  model = LogisticRegression(X, y, lr=0.1)
  activ_fn = Sigmoid()
  loss_fn = BCELoss()

  initial_ypred = activ_fn(model.forward())
  initial_loss = loss_fn(y, initial_ypred)

  model.train(activ_fn=activ_fn, loss_fn=loss_fn, epochs=500)

  final_ypred = activ_fn(model.forward())
  final_loss = loss_fn(y, final_ypred)

  assert final_loss < initial_loss

  # Convex BCE objective drives high confidence without saturation plateaus (>95%)
  accuracy = np.mean((final_ypred >= 0.5) == y)
  assert accuracy > 0.95


def test_logistic_regression_with_mse():
  """
  Verify LogisticRegression trains and converges with Sigmoid and MSELoss.
  """
  np.random.seed(42)

  X0 = np.random.randn(100, 2) - 1.5
  X1 = np.random.randn(100, 2) + 1.5
  X = np.vstack([X0, X1])
  y = np.vstack([np.zeros((100, 1)), np.ones((100, 1))])

  # lr=0.5 is intentionally higher than BCE (0.1): under MSE + Sigmoid, the
  # gradient retains y_pred*(1 - y_pred), which shrinks step size near 0 or 1.
  # Higher learning rate compensates for gradient attenuation over 500 epochs.
  model = LogisticRegression(X, y, lr=0.5)
  activ_fn = Sigmoid()
  loss_fn = MSELoss()

  initial_ypred = activ_fn(model.forward())
  initial_loss = loss_fn(y, initial_ypred)

  model.train(activ_fn=activ_fn, loss_fn=loss_fn, epochs=500)

  final_ypred = activ_fn(model.forward())
  final_loss = loss_fn(y, final_ypred)

  assert final_loss < initial_loss

  # MSE achieves lower confidence than BCE due to gradient shrinkage,
  # thresholded at >90%
  accuracy = np.mean((final_ypred >= 0.5) == y)
  assert accuracy > 0.90
