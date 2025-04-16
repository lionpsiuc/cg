/**
 * @file cg.c
 *
 * @brief Implementation of a serial conjugate gradient (CG) solver for the
 *        Poisson problem and dense linear systems.
 */

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * @brief Computes the dot product of two vectors.
 *
 * @param[in] x First vector.
 * @param[in] y Second vector.
 * @param[in] size Size of the vectors.
 *
 * @return The dot product value.
 */
double dot(double* x, double* y, int size) {
  double result = 0.0;
#pragma omp parallel for reduction(+ : result)
  for (int i = 0; i < size; i++) {
    result += x[i] * y[i];
  }
  return result;
}

/**
 * @brief Applies the dense matrix-vector product.
 *
 * Implements y=Ax where A is the dense matrix. This function computes the
 * product without explicitly storing the matrix.
 *
 * @param[in] x Input vector.
 * @param[out] y Output vector (i.e., Ax).
 * @param[in] N Size of the matrix.
 */
void        dense_matvec(double* x, double* y, int N) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    y[i] = 0.0;
    for (int j = 0; j < N; j++) {
      double A_ij = (double) (N - abs(i - j)) / N;
      y[i] += A_ij * x[j];
    }
  }
}

/**
 * @brief Creates the right-hand side vector for the dense linear system.
 *
 * @param[out] b Vector to be filled with right-hand side values.
 * @param[in] N Size of the vector.
 */
void        dense_rhs(double* b, int N) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    b[i] = 1.0;
  }
}

/**
 * @brief Writes the residual history to a file.
 *
 * Saves the residual norm at each iteration for convergence analysis.
 *
 * @param[in] residuals Array containing residual norms.
 * @param[in] iterations Number of iterations.
 * @param[in] filename Name of the output file.
 */
void write_residuals(double* residuals, int iterations, const char* filename) {
  FILE* fp = fopen(filename, "w");
  for (int i = 0; i <= iterations; i++) {
    fprintf(fp, "%d %e\n", i, residuals[i]);
  }
  fclose(fp);
}

/**
 * @brief Implements the CG algorithm for dense linear systems.
 *
 * Solves the linear system Ay=b using the CG method for the dense matrix. The
 * algorithm tracks residual history and uses the specified stopping criterion.
 *
 * @param[in] b Right-hand side vector.
 * @param[out] x Solution vector.
 * @param[in] N Size of the matrix.
 * @param[in] abstol Absolute tolerance for convergence.
 * @param[in] reltol Relative tolerance for convergence.
 * @param[in] max_iter Maximum number of iterations.
 * @param[out] iter_count Number of iterations performed.
 * @param[out] residuals Array to store residual norms at each iteration.
 */
void dense_cg(double* b, double* x, int N, double abstol, double reltol,
              int max_iter, int* iter_count, double* residuals) {
  double* r  = (double*) malloc(N * sizeof(double));
  double* p  = (double*) malloc(N * sizeof(double));
  double* Ap = (double*) malloc(N * sizeof(double));

  // Initialise solution to zeros
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    x[i] = 0.0;
  }

  // Calculate initial residual r_0=b-Ax_0=b
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    r[i] = b[i];
    p[i] = r[i];
  }

  double r0_norm = sqrt(dot(r, r, N));
  residuals[0]   = r0_norm; // Store initial residual norm

  // Stopping criterion threshold
  double threshold = fmax(reltol * r0_norm, abstol);

  int k;
  for (k = 0; k < max_iter; k++) {

    // Compute Ap
    dense_matvec(p, Ap, N);

    // a_k=(r_k,r_k)/(Ap_k,p_k)
    double rr    = dot(r, r, N);
    double pAp   = dot(p, Ap, N);
    double alpha = rr / pAp;

    // x_{k+1}=x_k+a_kp_k and r_{k+1}=r_k-a_kAp_k
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double r_norm    = sqrt(dot(r, r, N)); // Calculate current residual norm
    residuals[k + 1] = r_norm; // Store residual norm for this iteration

    // Check for convergence using the specified stopping criterion
    if (r_norm <= threshold) {
      k++; // Include this iteration in the count
      break;
    }

    // B_k=(r_{k+1},r_{k+1})/(r_k,r_k)
    double rr_new = dot(r, r, N);
    double beta   = rr_new / rr;

    // p_{k+1}=r_{k+1}+B_kp_k
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
      p[i] = r[i] + beta * p[i];
    }
  }

  // Free memory
  free(r);
  free(p);
  free(Ap);

  // Return iteration count
  *iter_count = k;
}

/**
 * @brief Runs the dense matrix convergence tests from section 3.3.
 */
void run_dense_matrix_tests() {
  int    N_values[] = {100, 1000, 10000};
  int    num_N      = sizeof(N_values) / sizeof(N_values[0]);
  int    max_iter   = 100000;      // Maximum number of iterations
  double abstol     = 0.0;         // Absolute tolerance
  double eps        = DBL_EPSILON; // Machine epsilon for double precision
  double reltol     = sqrt(eps);   // Relative tolerance
  printf("N\tIterations\tFinal Residual\tConvergence Factor\n");
  printf("----------------------------------------------------------\n");

  // Run the algorithm for each matrix size
  for (int i = 0; i < num_N; i++) {
    int N = N_values[i];

    // Allocate memory
    double* b         = (double*) malloc(N * sizeof(double));
    double* x         = (double*) malloc(N * sizeof(double));
    double* residuals = (double*) malloc((max_iter + 1) * sizeof(double));

    // Create right-hand side vector
    dense_rhs(b, N);

    // Run the dense CG algorithm
    int iter_count;
    dense_cg(b, x, N, abstol, reltol, max_iter, &iter_count, residuals);

    // Calculate convergence factor based on final iterations
    double conv_factor = 0.0;
    if (iter_count > 5) {
      // Use last few iterations to estimate convergence rate
      int start_idx = iter_count - 5 > 0 ? iter_count - 5 : 0;
      conv_factor   = pow(residuals[iter_count] / residuals[start_idx],
                          1.0 / (iter_count - start_idx));
    }

    printf("%d\t%d\t\t%.10e\t%.6f\n", N, iter_count, residuals[iter_count],
           conv_factor);

    // Write residual history to file
    char filename[50];
    sprintf(filename, "residuals-%d.dat", N);
    write_residuals(residuals, iter_count, filename);

    // Free memory
    free(b);
    free(x);
    free(residuals);
  }
}

/**
 * @brief Main function.
 *
 * Runs the dense matrix convergence test.
 *
 * @return 0 upon successful execution.
 */
int main() {
  run_dense_matrix_tests();
  return 0;
}
