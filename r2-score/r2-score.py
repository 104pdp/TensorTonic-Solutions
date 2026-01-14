import numpy as np

def r2_score(y_true, y_pred) -> float:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Constant-target edge case
    if np.all(y_true == y_true[0]):
        return 1.0 if np.all(y_pred == y_true) else 0.0

    # Sum of squared errors (SSE)
    sse = np.sum((y_true - y_pred) ** 2)

    # Total sum of squares (SST)
    y_mean = np.mean(y_true)
    sst = np.sum((y_true - y_mean) ** 2)

    return float(1.0 - sse / sst)
