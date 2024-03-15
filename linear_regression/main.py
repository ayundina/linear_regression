import numpy as np
from sklearn import datasets

from train import train
from predict import predict
from visualise import visualise_scatter
from visualise import visualise_prediction


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


def main():
    X, y = get_dataset("../data.csv")
    # X, y = generate_dataset()
    visualise_scatter(X.reshape(-1), y, "data_raw")
    model = train(X, y)
    y_prediction_line = predict(model, X)
    destandardized_error = model.mean_squared_error(X, y)
    print(f"DEstandardized_error = {destandardized_error}")
    visualise_prediction(X.reshape(-1), y, y_prediction_line, "data_trained")


if __name__ == "__main__":
    main()
