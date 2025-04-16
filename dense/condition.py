import numpy as np


def create_matrix(N):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i, j] = (N - abs(i - j)) / N
    return A


N = 10000
A = create_matrix(N)
eigenvalues = np.linalg.eigvalsh(A)
condition_number = max(eigenvalues) / min(eigenvalues)
print(f"Condition number for N = {N}: {condition_number}")
