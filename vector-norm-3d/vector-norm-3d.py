import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.asarray(v, dtype=float)

    # Single vector case: shape (3,)
    if v.ndim == 1:
        return float(np.sqrt(np.sum(v ** 2)))

    # Batch case: shape (N, 3)
    return np.sqrt(np.sum(v ** 2, axis=1))
