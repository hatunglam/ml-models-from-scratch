import numpy as np
from collections import Counter

def euclidian_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distance = [euclidian_dist(x, x_train) for  x_train in self.X_train]

        ids = np.argsort(distance)[:self.k]
        nearest_labels = [self.y_train[i] for i in ids]

        pred = Counter(nearest_labels).most_common()
        return pred[0][0]
    



