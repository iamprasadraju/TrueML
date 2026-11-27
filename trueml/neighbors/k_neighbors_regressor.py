import numpy as np


class KNeighborsRegressor:
  def __init__(self, X, y, n_neighbors = 3):
    self.X = X
    self.y = y
    self.n_neighbors = n_neighbors

  def predict(self, X_test, metric, method=None):
    distances = np.zeros(shape=(X_test.shape[0], self.X.shape[0]))
    # print(distances.shape)
    
    for i, test_sample in enumerate(X_test):
      for j, sample in enumerate(self.X):
        # calculate distance from every train sample to test sample
        distances[i][j] = metric(sample, test_sample)
        
    # sorted array indicies (row-wise sorted)
    idx = np.argsort(distances, axis=1)
    neighbors = self.y[idx][:, :self.n_neighbors]

    if method:
      return method(neighbors, axis=1)
    else:
      return neighbors
