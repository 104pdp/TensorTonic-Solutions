import numpy as np

def geometric_pmf_mean(k, p):

    k_arr = np.asarray(k, dtype=float)

    # PMF: p * (1-p)^(k-1) for k >= 1, else 0
    pmf = np.where(k_arr >= 1, p * np.power(1.0 - p, k_arr - 1.0), 0.0)
    pmf = np.asarray(pmf, dtype=float)  # ensure NumPy array

    mean = 1.0 / p

    return pmf, mean
