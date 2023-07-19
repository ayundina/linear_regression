def predict():
    try:
        if is_trained():
            read_input()
            get_weight_and_bias()
            calculate_prediction()
    except Exception as e:
        print(f"Error in predict():")
        print(f"{e}")


def is_trained():
    pass


def read_input():
    pass


def get_weight_and_bias():
    pass


def calculate_prediction():
    pass
