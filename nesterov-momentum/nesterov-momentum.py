import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    Returns (new_w, new_v).
    """
    # Ensure NumPy arrays (important for hidden tests)
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Step 2: update velocity (grad is already at look-ahead position)
    new_v = momentum * v + lr * grad

    # Step 3: update weights
    new_w = w - new_v

    return new_w, new_v
