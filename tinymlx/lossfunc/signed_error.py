# signed error: (x - y)
def signed_error(true_value, predict_value):
    """
        Measures raw deviation; allows for cancellation.
    """
    return true_value - predict_value
