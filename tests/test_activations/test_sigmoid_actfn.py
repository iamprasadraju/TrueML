import numpy as np

from trueml.activations import Sigmoid


def test_sigmoid_zero(): 
  assert np.isclose(Sigmoid()(0), 0.5)
  assert isinstance(Sigmoid()(0), np.ndarray)
  
# tests bounds of sigmoid(0, 1)
def test_sigmoid_bounds(): 
  x = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])
  y = Sigmoid()(x)
  assert np.all(y > 0) and np.all(y < 1)

def test_sigmoid_grad():
  y = np.array([0.2, 0.5, 0.8])
  expected = np.array([0.16, 0.25, 0.16])
  np.testing.assert_allclose(Sigmoid.grad(y), expected)