import numpy as np


def create_matrix(N):
    """
    Creates the requested dense matrix.

    Args:
        N (int): Size of the square matrix to create.

    Returns:
        numpy.ndarray: The constructed dense matrix.
    """
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i, j] = (N - abs(i - j)) / N
    return A


N = 1000
A = create_matrix(N)
eigenvalues = np.linalg.eigvalsh(A)
condition_number = max(eigenvalues) / min(eigenvalues)
print(f"Condition number for N = {N}: {condition_number}")
