import numpy as np


class LinearRegression():
    def __init__(self, lr=0.001):
        """
        Variables:
            n_iterations:   max number of steps to take in attempt to minimize error
            coef_:          coefficient. It determines the slope of the trend line
            bias:
        """
        self.lr = lr
        self.coef_ = None
        self.intercept_ = None

    def standardize_dataset(self, X, y):
        self.X_mean = X.mean(axis=0).reshape(-1)
        self.X_std = X.std(axis=0).reshape(-1)
        self.y_mean = y.mean()
        self.y_std = y.std()
        X = np.divide(np.subtract(X, self.X_mean), self.X_std)
        y = np.divide(np.subtract(y, self.y_mean), self.y_std)
        return X, y

    def destandardize_parameters(self):
        self.coef_ = np.divide(np.dot(self.coef_, self.y_std), self.X_std)
        self.intercept_ = np.subtract(self.y_mean, np.dot(self.coef_, self.X_mean))

    def predict(self, X):
        y_prediction = np.dot(X, self.coef_) + self.intercept_
        return y_prediction

    def score(self, X, y):
        y_predictions = self.predict(X)
        residual_sum_of_squares = np.sum((y_predictions - y) ** 2)
        total_sum_of_squares = np.sum((y.mean() - y) ** 2)
        coefficient_of_determination = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return coefficient_of_determination

    def fit(self, X, y):
        """
        Initialise weights and bias with zeros. Size of weights array equal to the number of features.
        1. Calculate prediction
        2. Calculate derivatives for weigt and bias
        3. Update weight and bias using derivatives and learning rate
        Variables:
            n_samples:      number of rows in a numpy matrix
            n_features:     number of columns in a numpy matrix
        """
        X, y = self.standardize_dataset(X, y)
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        derivative_coef = np.ones(n_features)
        derivative_intercept = 1
        while any(abs(i) > 0.001 for i in derivative_coef) or abs(derivative_intercept) > 0.001:
            y_predictions = self.predict(X)
            derivative_coef = (1 / n_samples) * np.dot(X.T, (y_predictions - y))
            derivative_intercept = (1 / n_samples) * np.sum(y_predictions - y)
            self.coef_ = self.coef_ - self.lr * derivative_coef
            self.intercept_ = self.intercept_ - self.lr * derivative_intercept
        self.destandardize_parameters()
