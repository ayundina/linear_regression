import matplotlib.pyplot as plt


def visualise_scatter(x, y, filename, save=True):
    plt.figure()
    plt.scatter(x, y)
    if save:
        plt.savefig(f"../visualisation/{filename}")


def visualise_prediction(x, y, y_prediction_line, filename):
    visualise_scatter(x, y, filename, save=False)
    plt.plot(x, y_prediction_line, color="black")
    plt.savefig(f"../visualisation/{filename}")
