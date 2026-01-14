import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)

    # Empty node
    if y.size == 0:
        return 0.0

    # Class counts
    _, counts = np.unique(y, return_counts=True)

    # Probabilities
    probs = counts / counts.sum()

    # Filter zero probabilities (numerical safety)
    probs = probs[probs > 0]

    # Entropy
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)
