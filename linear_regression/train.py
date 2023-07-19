def train():
    try:
        get_data()
        linear_regression()
        visualize()
        save_weights()
        calculate_precision()
    except Exception as e:
        print("Error in train()")
        print(f"{e}")


def visualize():
    visualize_data()
    visualize_model()
