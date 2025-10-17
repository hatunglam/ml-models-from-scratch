import numpy as np

# P(y) : Frequency of each class
# P(xi | y): Likelihood that the datapoint will look like that when it is class y -> Gaussian 
# => P(xi | y) = 1/(sqrt(2*pi*s**2*y)) * exp(-(xi - mu*y)/(2*s**2*y))

class NaiveBayes:
    
    def fit(self, X, y):
            n_samples, n_features = X.shape
            self._classes = np.unique(y) # Get all the unique classes
            


    def predict(self, X):
