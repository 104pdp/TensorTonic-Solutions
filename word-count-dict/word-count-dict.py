def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    counts = {}

    for sentence in sentences:
        for word in sentence:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

    return counts
