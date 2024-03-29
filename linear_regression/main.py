import numpy as np
from sklearn import datasets

from train import train
from predict import predict
from visualise import visualise_prediction
from visualise import visualise_features


def read_from_file(filename):
    data = np.genfromtxt(filename, unpack=True, delimiter=',', skip_header=True).T
    X = data[:, :-1]  # exclude last column
    y = data[:, -1]  # pick last column
    return X, y


def generate_dataset(n_samples, n_features, noise, random_state):
    X, y = datasets.make_regression(n_samples=24, n_features=1, noise=20, random_state=4)
    return X, y


def get_dataset(dataset_name):
    if dataset_name == "andrew_ng_ml_course_multivariate":
        X, y = read_from_file("../ex1data2.txt")
    elif dataset_name == "andrew_ng_ml_course_univariate":
        X, y = read_from_file("../ex1data1.txt")
    elif dataset_name == "codam_univariate":
        X, y = read_from_file("../data.csv")
    elif dataset_name == "sklearn_generate_univariate":
        X, y = datasets.make_regression(n_samples=24, n_features=1, noise=20, random_state=4)
    return X, y


def main():
    X, y = get_dataset("andrew_ng_ml_course_multivariate")
    visualise_features(X, y, "data_raw")
    model = train(X, y)
    y_prediction_line = predict(model, X)
    print(f"coefficient = {model.coef_}")
    print(f"intercept = {model.intercept_}")
    print(f"score = {model.score(X, y)}")
    print(f"y_prediction = {y_prediction_line}")
    visualise_prediction(X, y, y_prediction_line, "data_trained")


if __name__ == "__main__":
    main()
