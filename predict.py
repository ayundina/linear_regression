from linear_regression.data import get_data
import argparse
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.linear_regression import LinearRegression


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('km', help='input km to estimate price', type=int)
    args = parser.parse_args()
    return args.km


def predict(input_km):
    model = LinearRegression()
    print(
        f'1. Predicted price for {input_km} km is ${model.predict(input_km)}')
    return model


# def visualize(input_km):
#     km, price = get_data('dataset/data.csv')

#     intercept, coefficient = get_data('parameters/parameters.csv')

#     min_max_km = [min(min(km), input_km), max(max(km), input_km)]
#     model_graph = intercept + coefficient * min_max_km
#     prediction = intercept + coefficient * input_km
#     prediction[0] = round(prediction[0], 2)
#     print(f'Estimated price for {input_km} km is ${prediction[0]}')

#     plt.title('Car price / milage correlation')
#     plt.xlabel('milage - km')
#     plt.ylabel('price - $')

#     plt.scatter(input_km, prediction,
#                 label=f'estimated ${prediction[0]} for {input_km} km', marker='.', color='red')
#     plt.plot(min_max_km, model_graph, label='modeled data', linewidth=0.2,
#              color='black', dashes=[10, 10])
#     plt.scatter(km, price, label='training data', marker='.', color='black')
#     plt.legend()
#     plt.show()


if __name__ == '__main__':
    # input_km = parse()
    input_km = float(input('Enter km to estimate price: '))
    model = predict(input_km)
    model.visualize(input_km)
