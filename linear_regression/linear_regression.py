import csv
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.data import write_parameters


class LinearRegression:
    def __init__(self):
        self.dataset = 'dataset/data.csv'
        self.parameters = 'parameters/parameters.csv'
        self.km_list = []
        self.price_list = []
        self.learning_rate = 0.1
        self.intercept = 0.0
        self.coefficient = 0.0
        self.error_intercept = 0.0
        self.error_coefficient = 0.0
        self.list_error_intercept = []
        self.list_error_coefficient = []
        self.list_updated_intercept = []
        self.list_updated_coefficient = []


    def _get_dataset(self):
        with open(self.dataset, 'r') as dataset:
            reader = csv.reader(dataset)
            next(reader, None)
            for row in reader:
                self.km_list.append(float(row[0]))
                self.price_list.append(float(row[1]))
            self.km_array = np.array(self.km_list)
            self.price_array = np.array(self.price_list)


    def _get_parameters(self):
        with open(self.parameters, 'r') as parameters:
            reader = csv.reader(parameters)
            next(reader, None)
            intercept, coefficient = next(reader, [0, 0])
            self.intercept = float(intercept)
            self.coefficient = float(coefficient)



    def _fit_regression_line_original(self):
        estimate_price_array = self.intercept + self.coefficient * self.km_array

        error_intercept_array = estimate_price_array - self.price_array
        error_coefficient_array = (estimate_price_array - self.price_array) * self.km_array
        # self.log.write(f'sum error_intercept_array: {sum(error_intercept_array)}\n')
        # self.log.write(f'sum error_coefficient_array: {sum(error_coefficient_array)}\n')

        mean_error_intercept = sum(error_intercept_array) * ( 1 / self.size_dataset)
        mean_error_coefficient = sum(error_coefficient_array) * ( 1 / self.size_dataset)
        self.log.write(f'mean_error_intercept: {mean_error_intercept:.10f}\n')
        self.log.write(f'mean_error_coefficient: {mean_error_coefficient:.10f}\n')

        # step to take for convergence
        error_intercept = self.learning_rate * mean_error_intercept
        error_coefficient = self.learning_rate * mean_error_coefficient
        # self.log.write(f'error_intercept: {abs(error_intercept):.10f}\n')
        # self.log.write(f'error_coefficient: {abs(error_coefficient):.10f}\n')

        self.intercept = self.intercept - error_intercept
        self.coefficient = self.coefficient - error_coefficient
        self.list_updated_intercept.append(self.intercept)
        self.list_updated_coefficient.append(self.coefficient)
        self.list_error_intercept.append(error_intercept)
        self.list_error_coefficient.append(error_coefficient)
        self.log.write(f'UPDATED intercept: {self.intercept}\n')
        self.log.write(f'UPDATED coefficient: {self.coefficient}\n')

        # error_intercept = self.learning_rate * ((1/self.size_dataset) * sum((self.intercept + self.coefficient * self.km_array) - self.price_array))
        # error_coefficient = self.learning_rate * ((1/self.size_dataset) * sum(((self.intercept + self.coefficient * self.km_array) - self.price_array) * self.km_array))

        # self.intercept = self.intercept - error_intercept
        # self.coefficient = self.coefficient - error_coefficient

        # repeat until the step to convergence becomes too small
        ret = abs(error_intercept) < 0.000001 and abs(error_coefficient) < 0.000001
        self.log.write(f'{ret}\n')
        if ret == False:
            return False
        return True


    def _fit_regression_line_normalized(self):
        estimate_price_array = self.intercept + self.coefficient * self.km_array_normalized

        error_intercept_array = estimate_price_array - self.price_array_normalized
        error_coefficient_array = (estimate_price_array - self.price_array_normalized) * self.km_array_normalized
        # self.log.write(f'sum error_intercept_array: {sum(error_intercept_array)}\n')
        # self.log.write(f'sum error_coefficient_array: {sum(error_coefficient_array)}\n')

        mean_error_intercept = sum(error_intercept_array) * ( 1 / self.size_dataset)
        mean_error_coefficient = sum(error_coefficient_array) * ( 1 / self.size_dataset)
        self.log.write(f'mean_error_intercept: {mean_error_intercept:.10f}\n')
        self.log.write(f'mean_error_coefficient: {mean_error_coefficient:.10f}\n')

        # step to take for convergence
        error_intercept = self.learning_rate * mean_error_intercept
        error_coefficient = self.learning_rate * mean_error_coefficient
        # self.log.write(f'error_intercept: {abs(error_intercept):.10f}\n')
        # self.log.write(f'error_coefficient: {abs(error_coefficient):.10f}\n')

        self.intercept = self.intercept - error_intercept
        self.coefficient = self.coefficient - error_coefficient
        self.list_updated_intercept.append(self.intercept)
        self.list_updated_coefficient.append(self.coefficient)
        self.list_error_intercept.append(error_intercept)
        self.list_error_coefficient.append(error_coefficient)
        self.log.write(f'UPDATED intercept: {self.intercept}\n')
        self.log.write(f'UPDATED coefficient: {self.coefficient}\n')

        # error_intercept = self.learning_rate * ((1/self.size_dataset) * sum((self.intercept + self.coefficient * self.km_array_normalized) - self.price_array_normalized))
        # error_coefficient = self.learning_rate * ((1/self.size_dataset) * sum(((self.intercept + self.coefficient * self.km_array_normalized) - self.price_array_normalized) * self.km_array_normalized))

        # self.intercept = self.intercept - error_intercept
        # self.coefficient = self.coefficient - error_coefficient

        # repeat until the step to convergence becomes too small
        ret = abs(error_intercept) < 0.000001 and abs(error_coefficient) < 0.000001
        self.log.write(f'{ret}\n')
        if ret == False:
            return False
        return True


    def fit(self):
        self.log = open('train.log', 'w')
        self._get_dataset() 
        self.size_dataset = len(self.km_list)
        # self.km_array_normalized = (self.km_array - min(self.km_array)) / (max(self.km_array) - min(self.km_array))
        # self.price_array_normalized = (self.price_array - min(self.price_array)) / (max(self.price_array) - min(self.price_array))
        # self.km_array_normalized = (self.km_array - min(self.km_array)) / max(self.km_array)
        # self.price_array_normalized = (self.price_array - min(self.price_array)) / max(self.price_array)
        self.km_array_normalized = self.km_array / max(self.km_array)
        self.price_array_normalized = self.price_array / max(self.price_array)
        
        while self._fit_regression_line_normalized() is False:
            continue
        
        write_parameters(self.parameters, [self.intercept, self.coefficient])

        self.predict(1)
        _, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))


        mean_km = sum(self.km_array)/len(self.km_array)
        self.visualize_original_data(mean_km, ax[0][0])
        self.visualize_normalized_data(0.1, ax[0][1])
        self.visualize_gradient_descent(ax[0][2])
        self.visualize_original_and_normalized_together(0.1, ax[1][0])

        print(f'NORM intercept: {self.intercept}')
        print(f'NORM coefficient: {self.coefficient}')

        # mean_km = sum(self.km_array)/len(self.km_array)
        # mean_price = sum(self.price_array)/len(self.price_array)
        # mean_km_normed = (mean_km - min(self.km_array)) / (max(self.km_array) - min(self.km_array))
        # mean_price_normed = (mean_price - min(self.price_array)) / (max(self.price_array) - min(self.price_array))
        # print(f'FINAL interc: {(self.intercept + min(self.price_array)) * (min(self.price_array) - max(self.price_array))} = ({self.intercept} + {min(self.price_array)}) * ({min(self.price_array)} - {max(self.price_array)})')
        # print(f'FINAL coeff: {(self.coefficient + min(self.km_array)) * (min(self.km_array) - max(self.km_array))}')

        plt.show()






    def predict(self, x):
        self._get_parameters()
        self.prediction = self.intercept + self.coefficient * x
        return round(self.prediction, 2)


    def visualize_3d_gradient_descent(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Gradient Descent')
        ax.set_xlabel('intercept')
        ax.set_ylabel('coefficient')
        ax.set_zlabel('error')

        ax.plot(self.list_updated_intercept, self.list_updated_coefficient, self.list_error_intercept, label='gradient descent', marker='.', color='red')
        ax.legend()


    def visualize_gradient_descent(self, ax):
        ax.set_title('Gradient Descent')
        ax.set_xlabel('intercept and coefficient')
        ax.set_ylabel('error')

        ax.plot(self.list_updated_intercept, self.list_error_intercept, label='intercept gradient descent', marker='.', color='red')
        ax.plot(self.list_updated_coefficient, self.list_error_coefficient, label='coefficient gradient descent', marker='.', color='black')
        ax.legend()


    def visualize_original_data(self, input_km, ax):
        self._get_dataset()

        mean_price = sum(self.price_array)/len(self.price_array)
        mean_km = sum(self.km_array)/len(self.km_array)
        mean_price_norm = sum(self.price_array_normalized)/len(self.price_array_normalized)
        print(f'vertical shift = {mean_price - mean_price_norm}')
        # (self.intercept + self.coefficient * mean_km)
        self.original_intercept = self.intercept * max(self.price_array)
        #  + min(self.price_array)
        # self.original_coefficient = (self.coefficient) * (max(self.price_array)) / (max(self.km_array))
        # self.original_intercept = self.intercept + vertical_shift

        # self.original_coefficient = self.coefficient * (max(self.price_array) - min(self.price_array)) / (max(self.km_array) - min(self.km_array))
        self.original_coefficient = self.coefficient * (max(self.price_array) / max(self.km_array))

        print(f'max price / max km = {max(self.price_array)/max(self.km_array)} = {max(self.price_array)} / {max(self.km_array)}')
        print(f'min price = {min(self.price_array)}')
        print(f'min km = {min(self.km_array)}')
        print(f'self original intercept = {self.original_intercept}')
        print(f'self.original coefficient = {self.original_coefficient}')

        self.min_max_km = np.array([min(min(self.km_list), input_km), max(max(self.km_list), input_km)])
        self.model_graph = self.original_intercept + self.original_coefficient * self.min_max_km
        # print(f'original data model graph: {self.model_graph} = {self.original_intercept} + {vertical_shift} + {self.original_coefficient} * {self.min_max_km}')
        ax.set_title('Car price / milage correlation on original data')
        ax.set_xlabel('milage - km')
        ax.set_ylabel('price - $')

        self.prediction = self.original_intercept + self.original_coefficient * input_km
        # print(f'prediction on original data: {self.prediction} = {self.original_intercept} + {vertical_shift} + {self.original_coefficient} * {input_km}')
        ax.scatter(input_km, self.prediction,
                    label=f'estimated ${round(self.prediction, 2)} for {round(input_km, 2)} km', marker='.', color='red')

        ax.plot(self.min_max_km, self.model_graph, label='model', linewidth=0.2,
                 color='black', dashes=[10, 10])

        ax.scatter(self.km_list, self.price_list, label='training data',
                    marker='.', color='black')

        mean_km = sum(self.km_array)/len(self.km_array)
        mean_price = sum(self.price_array)/len(self.price_array)
        print(f'{mean_price} = original_intercept + original_coefficient * {mean_km}')

        ax.scatter(mean_km, mean_price, marker='+', color='red', label=f'training data mean. Price = {round(mean_price, 2)}, km = {round(mean_km, 2)}')


        mean_km_norm = sum(self.km_array_normalized)/len(self.km_array_normalized)
        mean_price_norm = sum(self.price_array_normalized)/len(self.price_array_normalized)

        # ax.scatter(mean_km_norm, mean_price_norm, marker='+', color='red', label=f'training data mean norm. Price = {round(mean_price_norm, 2)}, km = {round(mean_km_norm, 2)}')

        ax.legend()


    def visualize_normalized_data(self, input_km, ax):
        self.min_max_km = np.array([min(min(self.km_array_normalized), input_km), max(max(self.km_array_normalized), input_km)])
        self.model_graph = self.intercept + self.coefficient * self.min_max_km

        print(f'norm data model graph: {self.model_graph} = {self.intercept} + {self.coefficient} * {self.min_max_km}')

        ax.set_title('Car price / milage correlation on normalized data')
        ax.set_xlabel('milage - km')
        ax.set_ylabel('price - $')

        self.prediction = self.intercept + self.coefficient * input_km
        ax.scatter(input_km, self.prediction,
                    label=f'estimated ${round(self.prediction, 2)} for {input_km} km', marker='.', color='red')
        ax.plot(self.min_max_km, self.model_graph, label='model', linewidth=0.2,
                 color='black', dashes=[10, 10])
        ax.scatter(self.km_array_normalized, self.price_array_normalized, label='training data',
                    marker='.', color='black')

        mean_km = sum(self.km_array_normalized)/len(self.km_array_normalized)
        mean_price = sum(self.price_array_normalized)/len(self.price_array_normalized)
        print(f'mean price: {mean_price} = {self.intercept} + {self.coefficient} * {mean_km}')

        ax.scatter(mean_km, mean_price, marker='+', color='red', label='training data mean')
        ax.legend()

    def visualize_original_and_normalized_together(self, input_km, ax):
        self.visualize_original_data(input_km, ax)
        self.visualize_normalized_data(input_km, ax)