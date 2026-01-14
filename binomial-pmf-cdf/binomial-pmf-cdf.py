import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):

    # Safety: ensure valid inputs
    if k < 0 or k > n:
        return 0.0, 0.0

    # PMF
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    # CDF = sum_{i=0}^k PMF(i)
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    return float(pmf), float(cdf)
