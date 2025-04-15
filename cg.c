#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double f(double x1, double x2) {
  return 2.0 * M_PI * M_PI * sin(M_PI * x1) * sin(M_PI * x2);
}

void createMatrix(double* A, int N) {
  double h      = 1.0 / ((double) N + 1); // Grid spacing
  double h2_inv = 1.0 / (h * h);
  int    size   = N * N;

  // Initialise all elements to zero
  for (int i = 0; i < size * size; i++) {
    A[i] = 0.0;
  }

  // Set the diagonal and off-diagonal elements
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      int row = j * N + i; // Needed for row-major ordering
      A[row * size + row] =
          4.0 * h2_inv; // Diagonal element (i.e., coefficient of u(i,j))
      if (i > 0) {      // Connect to left neighbour, if not at the left edge
        A[row * size + (row - 1)] = -h2_inv;
      }
      if (i < N - 1) { // Connect to right neighbour, if not at the right edge
        A[row * size + (row + 1)] = -h2_inv;
      }
      if (j > 0) { // Connect to top neighbour, if not at the top edge
        A[row * size + (row - N)] = -h2_inv;
      }
      if (j < N - 1) { // Connect to bottom neighbour, if not at bottom edge
        A[row * size + (row + N)] = -h2_inv;
      }
    }
  }
}
