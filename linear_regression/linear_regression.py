import numpy as np


class LinearRegression():
    """
    Linear regression implementation from scratch.

    Attributes:
        lr (float): Learning rate for gradient descent.
        coef_ (numpy.ndarray): Coefficients (weights) of the linear regression model.
        intercept_ (float): Intercept of the linear regression model.
        X_mean (numpy.ndarray): Mean of each feature in the training dataset.
        X_std (numpy.ndarray): Standard deviation of each feature in the training dataset.
        y_mean (float): Mean of the target variable in the training dataset.
        y_std (float): Standard deviation of the target variable in the training dataset.

    Methods:
        standardize_dataset(X, y): Standardizes the features and target variable.
        destandardize_parameters(): Destandardizes the coefficients and intercept.
        predict(X): Predicts target variable for input samples.
        score(X, y): Computes the coefficient of determination (R^2) of the prediction.
        fit(X, y): Fits the linear regression model to the training dataset using gradient descent.
    """

    def __init__(self, lr=0.001):
        """
        Initializes the LinearRegression object.

        Args:
            lr (float, optional): Learning rate for gradient descent. Defaults to 0.001.
        """
        self.lr = lr
        self.coef_ = None
        self.intercept_ = None

    def standardize_dataset(self, X, y):
        """
        Standardizes the features and target variable.

        Args:
            X (numpy.ndarray): Features.
            y (numpy.ndarray): Target variable.

        Returns:
            tuple: Standardized features and target variable.
        """
        self.X_mean = X.mean(axis=0).reshape(-1)
        self.X_std = X.std(axis=0).reshape(-1)
        self.y_mean = y.mean()
        self.y_std = y.std()
        X = np.divide(np.subtract(X, self.X_mean), self.X_std)
        y = np.divide(np.subtract(y, self.y_mean), self.y_std)
        return X, y

    def destandardize_parameters(self):
        """
        Destandardizes the coefficients and intercept.
        """
        self.coef_ = np.divide(np.dot(self.coef_, self.y_std), self.X_std)
        self.intercept_ = np.subtract(self.y_mean, np.dot(self.coef_, self.X_mean))

    def predict(self, X):
        """
        Predicts target variable for input samples.

        Args:
            X (numpy.ndarray): Input samples.

        Returns:
            numpy.ndarray: Predicted target variable.
        """
        y_prediction = np.dot(X, self.coef_) + self.intercept_
        return y_prediction

    def score(self, X, y):
        """
        Computes the coefficient of determination (R^2) of the prediction.

        Args:
            X (numpy.ndarray): Features.
            y (numpy.ndarray): Target variable.

        Returns:
            float: Coefficient of determination (R^2).
        """
        y_predictions = self.predict(X)
        residual_sum_of_squares = np.sum((y_predictions - y) ** 2)
        total_sum_of_squares = np.sum((y.mean() - y) ** 2)
        coefficient_of_determination = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return coefficient_of_determination

    def fit(self, X, y):
        """
        Fits the linear regression model to the training dataset using gradient descent.

        Args:
            X (numpy.ndarray): Features.
            y (numpy.ndarray): Target variable.
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
