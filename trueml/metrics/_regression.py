import numpy as np


# mean absolute error
def mae(y_true, y_pred):
  return np.mean(np.abs(y_true - y_pred))

# mean squared error
def mse(y_true, y_pred):
  return np.mean(np.square(y_true - y_pred))

# root mean squared error
def rmse(y_true, y_pred):
  return np.sqrt(mse(y_true, y_pred))
