import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.asarray(y_train)

    # Edge case: empty test set
    n_test = len(X_test)
    if n_test == 0:
        return np.array([], dtype=int)

    # Find unique labels and their counts
    labels, counts = np.unique(y_train, return_counts=True)

    # Majority label: np.argmax is stable → first label in case of tie
    majority_label = labels[np.argmax(counts)]

    # Predict the same label for all test samples
    return np.full(n_test, majority_label, dtype=int)
