import numpy as np


def mean(x, axis=None): return np.mean(x, axis=axis)
def median(x, axis=None): return np.median(x, axis=axis)

def mode(x):
  values, counts = np.unique(x, return_counts=True)
  return values[np.argmax[counts]]
