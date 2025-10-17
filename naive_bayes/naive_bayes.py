import numpy as np

# P(y) : Frequency of each class
# P(xi | y): Likelihood that the datapoint will look like that when it is class y -> Gaussian 
# => P(xi | y) = 1/(sqrt(2*pi*s**2*y)) * exp(-(xi - mu*y)/(2*s**2*y))

class NaiveBayes:
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # Get all the unique classes
        n_classes = len(self._classes)

        # Calculate mean, variance, prior for each class
        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._priors = np.zeros(n_classes)

        for idx, cls in enumerate(self._classes):
            X_c = X[y==cls] # all datapoints belong to class C, shpae (n_samples X=c, n_features)
            self._mean[idx, :] = np.mean(X_c, axis=0) # Mean of all datapoints belongs to class C (all row C) 
            self._var[idx, :] = np.var(X_c, axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples) # Frequency of class C

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]  
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []

        # Calculate the posterior probability for each class
        for idx, cls in enumerate(self._classes):
            prior = np.log(self._priors[idx]) # prior of that class idx
            posterior = np.sum(np.log(self._pdf(idx, x))) # Likelihood of the datapoint x to that class idx
            posterior = posterior + prior # The numerator
            posteriors.append(posterior)

        # return classes with the highest posterior

