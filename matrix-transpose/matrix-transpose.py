import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    rows, cols = A.shape
    B = np.empty((cols, rows), dtype=A.dtype)

    for row in range(rows):
        for col in range(cols):
            B[col, row] = A[row, col]

    return B
