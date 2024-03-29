def predict(model, X):
    """
    Predicts labels for input samples using a trained model.

    Args:
        model (LinearRegression): A trained linear regression model.
        X (numpy.ndarray): An array of input samples, where each row represents a sample and each column represents a feature.

    Returns:
        numpy.ndarray: An array of predicted labels corresponding to the input samples.
    """
    y_prediction = model.predict(X)
    return y_prediction
