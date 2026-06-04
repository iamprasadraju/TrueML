import numpy as np
from tinymlx.activations import sigmoid

class LogisticRegression:
    def __init__(self, n_features, lr=0.01):
        self.lr = lr
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    def forward(self, X):
        z = X @ self.weights + self.bias
        return sigmoid(z)

    def backward(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db