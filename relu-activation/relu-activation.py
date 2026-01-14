import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x, dtype=float)
    y = np.maximum(0.0, x)
    return np.atleast_1d(y)
