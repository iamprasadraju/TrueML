# signed error: (x - y)

class SignedError:   
    def __call__(self, true_value, predict_value):
        """
            Measures raw deviation; allows for cancellation.
        """
        return true_value - predict_value
