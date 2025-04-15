#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Dot product
double dot(double* x, double* y, int size) {
  double result = 0.0;
  for (int i = 0; i < size; i++) {
    result += x[i] * y[i];
  }
  return result;
}

// Function given in assignment
double f(double x1, double x2) {
  return 2.0 * M_PI * M_PI * sin(M_PI * x1) * sin(M_PI * x2);
}

// Finite difference method
void fd(double* A, int N) {
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
      int index = j * N + i; // Needed for row-major ordering

      // Diagonal element (i.e., coefficient of u(i,j))
      A[index * size + index] = 4.0 * h2_inv;

      // Connect to left neighbour, if not at the left edge
      if (i > 0) {
        A[index * size + (index - 1)] = -h2_inv;
      }

      // Connect to right neighbour, if not at the right edge
      if (i < N - 1) {
        A[index * size + (index + 1)] = -h2_inv;
      }

      // Connect to top neighbour, if not at the top edge
      if (j > 0) {
        A[index * size + (index - N)] = -h2_inv;
      }

      // Connect to bottom neighbour, if not at bottom edge
      if (j < N - 1) {
        A[index * size + (index + N)] = -h2_inv;
      }
    }
  }
}

// Matrix-vector multiplication
void mvp(double* A, double* x, double* y, int N) {
  int size = N * N;

  // Initialise with zeros
  for (int i = 0; i < size; i++) {
    y[i] = 0.0;
  }
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      y[i] += A[i * size + j] * x[j];
    }
  }
}

// Optimised method of calculting Ax; instead of storing A, we multiply directly
// using this function
void poisson(double* x, double* y, int N) {
  double h      = 1.0 / ((double) N + 1); // Grid spacing
  double h2_inv = 1.0 / (h * h);

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      int index = j * N + i; // Needed for row-major ordering

      // Diagonal term: multiplying by 4/h²
      y[index] = 4.0 * h2_inv * x[index];

      // Connect to left neighbour, if not at the left edge
      if (i > 0) {
        y[index] -= h2_inv * x[index - 1];
      }

      // Connect to right neighbour, if not at the right edge
      if (i < N - 1) {
        y[index] -= h2_inv * x[index + 1];
      }

      // Connect to top neighbour, if not at the top edge
      if (j > 0) {
        y[index] -= h2_inv * x[index - N];
      }

      // Connect to bottom neighbour, if not at bottom edge
      if (j < N - 1) {
        y[index] -= h2_inv * x[index + N];
      }
    }
  }
}

// Method to calculate the right-hand side vector as per the five-point stencil
// approximation
void rhs(double* b, int N, double (*f)(double, double)) {
  double h = 1.0 / ((double) N + 1);
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      int    index = j * N + i;
      double x1    = (i + 1) * h; // The x-coordinate of the grid point
      double x2    = (j + 1) * h; // The y-coordinate of the grid point
      b[index]     = f(x1, x2);   // Evaluate our function at this point
    }
  }
}

// Just testing to make sure my poisson function is correctly multiplying things
// together; comparing it with an explicit matrix-vector multiplication using
// mvp
void test() {
  int     N    = 3;
  int     size = N * N;
  double* A    = (double*) malloc(size * size * sizeof(double));
  double* x    = (double*) malloc(size * sizeof(double));
  double* y1   = (double*) malloc(size * sizeof(double));
  double* y2   = (double*) malloc(size * sizeof(double));
  fd(A, N);
  for (int i = 0; i < size; i++) {
    x[i] = 1.0;
  }
  mvp(A, x, y1, N);
  poisson(x, y2, N);
  for (int i = 0; i < size; i++) {
    printf("%f\n", y1[i]);
  }
  for (int i = 0; i < size; i++) {
    printf("%f\n", y2[i]);
  }
  free(A);
  free(x);
  free(y1);
  free(y2);
}

int main() {
  int     N    = 3;
  int     size = N * N;
  double* b    = (double*) malloc(size * sizeof(double));
  rhs(b, N, f);
  for (int i = 0; i < size; i++) {
    printf("%6.4f\n", b[i]);
  }
  test();
  free(b);
  return 0;
}
