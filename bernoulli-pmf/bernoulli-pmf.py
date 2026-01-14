import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    x = np.asarray(x)

    # PMF: p for x==1, (1-p) for x==0
    pmf = np.where(x == 1, p, 1.0 - p).astype(float)

    mean = float(p)
    var = float(p * (1.0 - p))

    return pmf, mean, var
