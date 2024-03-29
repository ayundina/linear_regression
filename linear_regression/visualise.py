import matplotlib.pyplot as plt


def visualise_scatter(x, y, filename, save=True):
    plt.figure()
    plt.scatter(x, y)
    if save:
        plt.savefig(f"../visualisation/{filename}")


def visualise_features(X, y, filename):
    for i, feature in enumerate(X.T):
        visualise_scatter(feature.reshape(-1), y, f"{filename}_{i}")


def visualise_prediction(X, y, y_prediction, filename):
    for i, feature in enumerate(X.T):
        visualise_scatter(feature.reshape(-1), y, f"{filename}_{i}", save=False)
        plt.plot(feature, y_prediction, color="tomato", linewidth=0.5)
        plt.savefig(f"../visualisation/{filename}_{i}")
