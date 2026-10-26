import numpy as np

from trueml.activations import Linear


# test scalar input and list input (ArrayLike) returns nparray
def test_linear_actfn_scalar(): result = Linear()(1); assert result == 1; assert isinstance(result, np.ndarray)
def test_linear_actfn_list(): result = Linear()([1, 2, 3, 4]); np.testing.assert_array_equal(result, [1, 2, 3, 4])
def test_linear_actfn_nparray(): x = np.random.rand(3, 4); np.testing.assert_array_equal(Linear()(x), x)

# test for grad (all ones)
def test_linear_grad():
  x = np.random.rand(3, 4)
  np.testing.assert_array_equal(Linear.grad(x), np.ones_like(x))

