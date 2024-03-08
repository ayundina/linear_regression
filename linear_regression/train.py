from linear_regression import LinearRegression


def train(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model
