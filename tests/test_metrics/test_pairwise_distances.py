import numpy as np

from trueml.metrics import euclidean


# test euclidean distance (pairwise) 1d, 2d, nd
def test_euclidean_1d():
  x = np.random.randint(1, 9, size=(1, 1))
  y = np.random.randint(1, 9, size=(1, 1))
  assert euclidean(x, y) == np.linalg.norm(x - y)

def test_euclidean_2d():
  x = np.random.randint(1, 9, size=(1, 2))
  y = np.random.randint(1, 9, size=(1, 2))
  assert euclidean(x, y) == np.linalg.norm(x - y)

def test_euclidean_3d():
  x = np.random.randint(1, 9, size=(1, 3))
  y = np.random.randint(1, 9, size=(1, 3))
  assert euclidean(x, y) == np.linalg.norm(x - y)

if __name__ == "__main__":
  test_euclidean_1d()
