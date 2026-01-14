import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Ensure 1D inputs
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")

    # Ensure same length
    if x.shape[0] != y.shape[0]:
        raise ValueError("Vectors must have the same length")

    return float(np.dot(x, y))
