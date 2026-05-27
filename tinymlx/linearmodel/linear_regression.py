import numpy as np


class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None

        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        self.X =  X
        self.y = y
        
        if self.X.shape[0] == self.y.shape[0]:
            self.weights = np.random.rand(self.X.shape[1])
            self.bias = 0
        else:
            raise ValueError

    def train(self):
        y_pred = (self.X @ self.weights) + self.bias
        return y_pred
