import numpy as np

class LinearRegression:
    def __init__(self, n_features, lr = 0.01):
        # lr - learning rate
        self.lr = lr 
        # number of features X_train.shape[1] (columns)
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0.0

    def forward(self, X_train):
        # y_pred 
        return X_train @ self.weights + self.bias
        
    def backward(self, dw, db):
            # Manually apply the update rule using the stored learning rate
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
