"""Tests for sigmoid activation."""
from trueml.activations.sigmoid import sigmoid

class TestSigmoid:
    def test_sigmoid_basic(self):
        result = np.array(sigmoid(np.array([0, 1, -1], dtype=float)))
        expected = np.array([0.5, 0.73105858, 0.26894142], dtype=float)
        np.testing.assert_allclose(result, expected, 1e-5)
