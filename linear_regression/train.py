from linear_regression import LinearRegression
# from sklearn.linear_model import LinearRegression


def train(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model
