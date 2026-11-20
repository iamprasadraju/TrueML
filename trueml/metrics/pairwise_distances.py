import numpy as np


# https://en.wikipedia.org/wiki/Euclidean_distance
# similar to numpy: np.linalg.norm(p - q)
def euclidean(p, q): return np.sqrt(np.sum(np.square(p - q)))
