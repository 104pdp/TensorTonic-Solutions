import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.asarray(v, dtype=float)
    eps = 1e-10

    # Single vector (3,)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > eps else np.zeros_like(v)

    # Batch (N, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)

    out = np.zeros_like(v)
    np.divide(v, norms, out=out, where=(norms > eps))  # no invalid division
    return out
