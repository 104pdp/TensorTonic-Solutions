import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels (single-label).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    n = yt.size
    if n == 0:
        return 0.0  # на всякий случай, хотя по условию обычно n>0

    tp = int(np.sum(yt == yp))  # количество совпадений
    # micro-F1 для single-label multiclass = TP / N
    return float(tp / n)
