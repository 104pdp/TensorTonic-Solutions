import numpy as np

def relu(x):

    x_arr = np.asarray(x, dtype=float)
    y = np.maximum(0.0, x_arr)

    # Если скаляр — вернуть массив формы (1,)
    if y.ndim == 0:
        return np.array([y], dtype=float)

    return y
