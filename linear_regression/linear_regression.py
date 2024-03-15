import numpy as np

from visualise import visualise_scatter
from visualise import visualise_prediction


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

    def standardize_feature(self, feature):
        scaler = StandardScaler()
        scaled_feature = scaler.fit_transform(feature)
        return scaled_feature, scaler

    def standardize_dataset(self, X, y):
        self.X_mean = X.mean(0).reshape(-1)
        self.X_std = X.std(0).reshape(-1)
        self.y_mean = y.mean()
        self.y_std = y.std()
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std
        return X, y

    def destandardize_parameters(self):
        print(f"standardized_weights = {self.weights}")
        print(f"standardized_bias = {self.bias}")
        self.weights = self.weights * self.y_std / self.X_std
        self.bias = self.y_mean - self.weights[0] * self.X_mean[0]
        print(f"DEstandardized_weights = {self.weights}")
        print(f"DEstandardized_bias = {self.bias}")

    def predict(self, X, standardize_features=False):
        y_prediction = np.dot(X, self.weights) + self.bias
        return y_prediction

    def mean_squared_error(self, X, y):
        prediction = self.predict(X, standardize_features=False)
        error = np.mean((y - prediction) ** 2)
        return error

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
        print(f"X = {X.reshape(-1)}")
        print(f"y = {y}")
        X, y = self.standardize_dataset(X, y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        print(f"n_features = {n_features}")
        print(f"n_samples = {n_samples}")
        self.bias = 0
        derivative_weights = [1]
        derivative_bias = 1
        while abs(derivative_weights[0]) > 0.1 or abs(derivative_bias) > 0.1:
            y_predictions = np.dot(X, self.weights) + self.bias
            derivative_weights = (1 / n_samples) * \
                np.dot(X.T, (y_predictions - y))
            derivative_bias = (1 / n_samples) * np.sum(y_predictions - y)
            self.weights = self.weights - self.lr * derivative_weights
            self.bias = self.bias - self.lr * derivative_bias

        # print(f"derivative_weights = {derivative_weights}")
        # print(f"derivative_bias = {derivative_bias}")

        standardized_error = self.mean_squared_error(X, y)
        print(f"standardized_error = {standardized_error}")

        # y_prediction_line = self.predict(X, standardize_features=False)
        # filename = "data_trained_standardized"
        # visualise_prediction(X.reshape(-1), y, y_prediction_line, filename)

        self.destandardize_parameters()
