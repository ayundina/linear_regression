import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from train import train
from predict import predict


def get_dataset(filename):
    data = np.genfromtxt(filename, unpack=True,
                         delimiter=',', skip_header=True).T
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return X, y


def generate_dataset():
    X, y = datasets.make_regression(
        n_samples=24, n_features=1, noise=20, random_state=4)
    return X, y


def visualise_scatter(x, y):
    plt.scatter(x, y)
    plt.savefig("../visualisation/data")


def visualise_prediction(x, y, y_prediction_line):
    plt.plot(x, y_prediction_line, color="black")
    visualise_scatter(x, y)


def main():
    X, y = get_dataset("../data.csv")
    # X, y = generate_dataset()
    visualise_scatter(X.reshape(-1), y)
    model = train(X, y)
    y_prediction_line = predict(model, X)
    visualise_prediction(X.reshape(-1), y, y_prediction_line)


if __name__ == "__main__":
    main()
