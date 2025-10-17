import numpy as np

class LinearRegression:
    def __init__(self, lr=1e-3, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias

            dW = (X.T @ (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples

            self.weights = self.weights - self.lr * dW
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        pred = X @ self.weights + self.bias 
        return pred 
    
def mse(y_pred, y):
    return np.mean((y_pred - y)**2)
