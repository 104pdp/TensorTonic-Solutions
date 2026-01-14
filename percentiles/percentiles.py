import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)

    # NumPy >= 1.22
    try:
        return np.percentile(x, q, method="linear")
    except TypeError:
        # NumPy < 1.22 (старый API)
        return np.percentile(x, q, interpolation="linear")
