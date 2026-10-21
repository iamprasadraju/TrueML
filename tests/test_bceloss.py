import numpy as np

from trueml.losses import BCELoss


def test_bce_loss_forward():
  """
  Verify BCELoss forward value against analytical calculation.
  """
  # Symmetric 2-sample case where both samples yield identical loss -ln(0.8)
  y_true = np.array([[1.0], [0.0]])
  y_pred = np.array([[0.8], [0.2]])

  loss_fn = BCELoss()
  loss = loss_fn(y_true, y_pred)

  expected = -np.log(0.8)
  np.testing.assert_allclose(loss, expected, rtol=1e-5)


def test_bce_loss_grad_numerical():
  """
  Verify BCELoss analytical gradient against finite differences.
  """
  y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
  y_pred = np.array([[0.7], [0.3], [0.8], [0.1]])

  loss_fn = BCELoss()
  analytic_grad = loss_fn.grad(y_true, y_pred)

  # Uses two-sided central difference (f(x+h) - f(x-h)) / (2h):
  # Truncation error is O(h^2) instead of one-sided O(h), preventing false divergence
  # between analytical calculus and finite differences at float64.
  eps = 1e-6
  num_grad = np.zeros_like(y_pred)
  for i in range(y_pred.shape[0]):
    y_plus = y_pred.copy()
    y_plus[i, 0] += eps
    y_minus = y_pred.copy()
    y_minus[i, 0] -= eps

    num_grad[i, 0] = (loss_fn(y_true, y_plus) - loss_fn(y_true, y_minus)) / (2 * eps)

  np.testing.assert_allclose(analytic_grad, num_grad, rtol=1e-5, atol=1e-5)


def test_bce_loss_numerical_stability():
  """
  Verify BCELoss handles saturated probabilities (0 and 1) without NaN/inf.
  """
  # Boundary edge case: completely wrong and fully saturated predictions.
  # Mathematically ln(0) = -inf and denominator y_pred*(1 - y_pred) = 0.
  # Tests that internal epsilon-clipping keeps forward loss and gradient finite.
  y_true = np.array([[1.0], [0.0]])
  y_pred = np.array([[0.0], [1.0]])

  loss_fn = BCELoss()
  loss = loss_fn(y_true, y_pred)
  grad = loss_fn.grad(y_true, y_pred)

  assert not np.isnan(loss)
  assert not np.isinf(loss)
  assert not np.any(np.isnan(grad))
  assert not np.any(np.isinf(grad))
