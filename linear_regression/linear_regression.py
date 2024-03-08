import numpy as np


class LinearRegression():
    def __init__(self, lr=0.01, n_iterations=1000):
        """
        Variables:
            lr:             learning rate. Used in the gradient descent to help define size of a step to make to minimaze the error
            n_iterations:   max number of steps to take in attempt to minimize error
            weigt:          coefficient. It determines the slope of the trend line
            bias:
        """
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

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
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        print(f"X = {X}")
        print(f"y = {y}")
        print(f"n_features = {n_features}")
        print(f"n_samples = {n_samples}")
        self.bias = 0
        derivative_weights = [1]
        derivative_bias = 1
        # for _ in range(self.n_iterations):
        while abs(derivative_weights[0]) > 0.1 or abs(derivative_bias) > 0.1:
            y_predictions = np.dot(X, self.weights) + self.bias
            derivative_weights = (1 / n_samples) * \
                np.dot(X.T, (y_predictions - y))
            derivative_bias = (1 / n_samples) * np.sum(y_predictions - y)
            print(f"derivative_weights = {derivative_weights}")
            print(f"derivative_bias = {derivative_bias}")
            if np.isnan(derivative_weights[0]) and np.isnan(derivative_bias):
                print("breaking")
                break
            self.weights = self.weights - self.lr * derivative_weights
            self.bias = self.bias - self.lr * derivative_bias
        print(f"derivative_weights = {derivative_weights}")
        print(f"derivative_bias = {derivative_bias}")

    def predict(self, X):
        y_prediction = np.dot(X, self.weights) + self.bias
        return y_prediction
