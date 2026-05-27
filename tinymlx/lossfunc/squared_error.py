# squared error: (x - y) ** 2

def squared_error(true_value, predicted_value):
    """
        Penalizes large errors exponentially more than small ones.
    """
    return (true_value - predicted_value) ** 2
    