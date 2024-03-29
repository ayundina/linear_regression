import matplotlib.pyplot as plt


def visualise_scatter(x, y, filename, save=True):
    """
    Visualizes a scatter plot of data points.

    Args:
        x (numpy.ndarray): Array of x-coordinates.
        y (numpy.ndarray): Array of y-coordinates.
        filename (str): The name of the file to save the visualization.
        save (bool, optional): If True, saves the visualization as a file. Defaults to True.

    Returns:
        None
    """
    plt.figure()
    plt.scatter(x, y)
    if save:
        plt.savefig(f"../visualisation/{filename}")


def visualise_features(X, y, filename):
    """
    Visualizes each feature in the dataset against the target variable.

    Args:
        X (numpy.ndarray): An array of features, where each row represents a sample and each column represents a feature.
        y (numpy.ndarray): An array of labels corresponding to the samples.
        filename (str): The base name for saving the visualizations. Each feature will be appended with an index.

    Returns:
        None
    """
    for i, feature in enumerate(X.T):
        visualise_scatter(feature.reshape(-1), y, f"{filename}_{i}")


def visualise_prediction(X, y, y_prediction, filename):
    """
    Visualizes the actual and predicted values against each feature.

    Args:
        X (numpy.ndarray): An array of features, where each row represents a sample and each column represents a feature.
        y (numpy.ndarray): An array of actual labels corresponding to the samples.
        y_prediction (numpy.ndarray): An array of predicted labels corresponding to the samples.
        filename (str): The base name for saving the visualizations. Each feature will be appended with an index.

    Returns:
        None
    """
    for i, feature in enumerate(X.T):
        visualise_scatter(feature.reshape(-1), y, f"{filename}_{i}", save=False)
        plt.plot(feature, y_prediction, color="tomato", linewidth=0.5)
        plt.savefig(f"../visualisation/{filename}_{i}")
