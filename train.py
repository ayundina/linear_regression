# from sklearn.linear_model import LinearRegression
from linear_regression.linear_regression import LinearRegression
from linear_regression.data import get_data, write_parameters


# def train_model():
#     km, price = get_data('dataset/data.csv')

#     model = LinearRegression()
#     model.fit(km.reshape(-1, 1), price)
#     print(km.reshape(-1, 1))
#     print(model.score(km.reshape(-1, 1), price))

#     intercept = float(model.intercept_)
#     coefficient = float(model.coef_[0])
#     print('Intercept:', intercept)
#     print('Coefficient:', coefficient)

#     write_parameters('./parameters/parameters.csv',
#                      [intercept, coefficient])


def train_model():
    model = LinearRegression()
    model.fit()
    # model.visualize_gradient_descent()


if __name__ == '__main__':
    train_model()
