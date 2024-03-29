from linear_regression import LinearRegression
# from sklearn.linear_model import LinearRegression


def train(X, y):
    """
    Trains a linear regression model on the provided dataset.

    Args:
        X (numpy.ndarray): An array of features, where each row represents a sample and each column represents a feature.
        y (numpy.ndarray): An array of labels corresponding to the samples.

    Returns:
        LinearRegression: A trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model
