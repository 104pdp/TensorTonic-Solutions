import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    Returns: (mean, median, mode) as floats
    """
    x = np.asarray(x, dtype=float)

    mean_val = float(np.mean(x))
    median_val = float(np.median(x))

    counts = Counter(x)
    max_freq = max(counts.values())
    mode_val = float(min(v for v, c in counts.items() if c == max_freq))

    return mean_val, median_val, mode_val
