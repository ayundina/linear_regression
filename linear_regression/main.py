import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression

from train import train
from predict import predict
from visualise import visualise_scatter
from visualise import visualise_prediction


def get_dataset(filename):
    data = np.genfromtxt(filename, unpack=True, delimiter=',', skip_header=True).T
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return X, y


def generate_dataset():
    X, y = datasets.make_regression(
        n_samples=24, n_features=1, noise=20, random_state=4)
    return X, y


def main():
    # X, y = get_dataset("../data.csv")
    X, y = get_dataset("../ex1data1.txt")
    # X, y = generate_dataset()
    visualise_scatter(X.reshape(-1), y, "data_raw")
    model = train(X, y)
    y_prediction_line = predict(model, X)
    print(f"weights = {model.weights}")
    print(f"bias = {model.bias}")
    print(f"score = {model.score(X, y)}")
    visualise_prediction(X.reshape(-1), y, y_prediction_line, "data_trained")

    model = LinearRegression()
    model.fit(X, y)
    print(f"score = {model.score(X, y)}")
    print(f"weights = {model.coef_}")
    print(f"bias = {model.intercept_}")
    visualise_prediction(X.reshape(-1), y, model.predict(X), "data_skl_trained")


if __name__ == "__main__":
    main()

# next step: implement for multivariate
