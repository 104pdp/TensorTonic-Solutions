import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return np.tanh(x)
