import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Convert inputs to NumPy arrays (IMPORTANT for hidden tests)
    x_t = np.asarray(x_t, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    Wx = np.asarray(Wx, dtype=float)
    Wh = np.asarray(Wh, dtype=float)
    b = np.asarray(b, dtype=float)

    pre_act = x_t @ Wx + h_prev @ Wh + b
    h_t = np.tanh(pre_act)

    return h_t
