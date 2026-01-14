import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Map vocab word -> index
    vocab_index = {word: i for i, word in enumerate(vocab)}

    # Initialize zero vector
    bow = np.zeros(len(vocab), dtype=int)

    # Count tokens
    for token in tokens:
        if token in vocab_index:
            bow[vocab_index[token]] += 1

    return bow
