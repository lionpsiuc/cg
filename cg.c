/**
 * @file cg.c
 *
 * @brief Implementation of a serial conjugate gradient (CG) solver for the
 *        Poisson problem.
 *
 * This programme solves the Poisson equation on a unit square with Dirichlet
 * boundary conditions using the CG method.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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
  for (int i = 0; i < size; i++) {
    result += x[i] * y[i];
  }
  return result;
}

/**
 * @brief Right-hand side function for the Poisson problem.
 *
 * @param[in] x1 First coordinate.
 * @param[in] x2 Second coordinate.
 *
 * @return Value of the function at the given coordinates.
 */
double f(double x1, double x2) {
  return 2.0 * M_PI * M_PI * sin(M_PI * x1) * sin(M_PI * x2);
}

/**
 * @brief Applies the discrete Poisson operator (i.e., a matrix-vector product).
 *
 * Implements the matrix-vector product y=Ax where A is the discrete Laplacian
 * matrix using the standard five-point stencil. This function avoids explicitly
 * storing the matrix A.
 *
 * @param[in] x Input vector.
 * @param[out] y Output vector (i.e., Ax).
 * @param[in] N Number of grid points in each dimension.
 */
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

/**
 * @brief Constructs the right-hand side vector for the linear system.
 *
 * Populates the vector b with values of the function f evaluated at grid
 * points.
 *
 * @param[out] b Vector to be filled with right-hand side values.
 * @param[in] N Number of grid points in each dimension.
 * @param[in] f Pointer to function evaluating the right-hand side.
 */
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

/**
 * @brief Writes the numerical solution to a file.
 *
 * Saves the computed solution u along with the coordinates of each grid point
 * to a file for visualisation and analysis.
 *
 * @param[in] u Vector containing the solution values.
 * @param[in] N Number of grid points in each dimension.
 * @param[in] filename Name of the output file.
 */

void write_solution(double* u, int N, const char* filename) {
  double h  = 1.0 / ((double) N + 1);
  FILE*  fp = fopen(filename, "w");
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      int    index   = j * N + i;
      double x_coord = (i + 1) * h;
      double y_coord = (j + 1) * h;
      fprintf(fp, "%f %f %f\n", x_coord, y_coord, u[index]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

/**
 * @brief Implements the CG algorithm.
 *
 * Solves the linear system Ax=b using the CG method where A is the discrete
 * Laplacian. The algorithm terminates when either the maximum number of
 * iterations is reached or the residual norm falls below the specified
 * tolerance.
 *
 * @param[in] b Right-hand side vector.
 * @param[out] x Solution vector.
 * @param[in] N Number of grid points in each dimension.
 * @param[in] tol Convergence tolerance.
 * @param[in] max_iter Maximum number of iterations.
 * @param[out] iter_count Number of iterations performed.
 * @param[out] final_residual Final residual norm.
 */
void cg(double* b, double* x, int N, double tol, int max_iter, int* iter_count,
        double* final_residual) {
  int size = N * N;

  // Allocate memory for vectors
  double* r  = (double*) malloc(size * sizeof(double));
  double* p  = (double*) malloc(size * sizeof(double));
  double* Ap = (double*) malloc(size * sizeof(double));

  // Initialise to ones (i.e., our initial guess)
  for (int i = 0; i < size; i++) {
    x[i] = 1.0;
  }

  // Calculate initial residual r_0=b-Ax_0
  poisson(x, Ap, N); // Reusing Ap to store Ax_0 temporarily
  for (int i = 0; i < size; i++) {
    r[i] = b[i] - Ap[i];
    p[i] = r[i];
  }

  double rr               = dot(r, r, size);
  double initial_residual = sqrt(rr);
  int    k;
  for (k = 0; k < max_iter; k++) {

    // Check for convergence
    double residual_norm = sqrt(rr);
    if (residual_norm <= tol * initial_residual || residual_norm <= tol) {
      break;
    }

    // Compute Ap
    poisson(p, Ap, N);

    // a_k=(r_k,r_k)/(Ap_k,p_k)
    double pAp   = dot(p, Ap, size);
    double alpha = rr / pAp;

    // x_{k+1}=x_k+a_kp_k and r_{k+1}=r_k-a_kAp_k
    for (int i = 0; i < size; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    // Calculate the new rr for B_k
    double rr_new = dot(r, r, size);

    // B_k=(r_{k+1},r_{k+1})/(r_k,r_k)
    double beta = rr_new / rr;

    // Update rr for the next iteration
    rr = rr_new;

    // p_{k+1}=r_{k+1}+B_kp_k
    for (int i = 0; i < size; i++) {
      p[i] = r[i] + beta * p[i];
    }
  }

  // Free memory
  free(r);
  free(p);
  free(Ap);

  // Return results
  *iter_count     = k;
  *final_residual = sqrt(rr);
}

/**
 * @brief Computes the analytical solution to the Poisson problem.
 *
 * Returns the exact solution to the Poisson equation with zero Dirichlet
 * boundary conditions.
 *
 * @param[in] x1 First coordinate.
 * @param[in] x2 Second coordinate.
 *
 * @return Exact solution value at the given coordinates.
 */
double analytical_solution(double x1, double x2) {
  return sin(M_PI * x1) * sin(M_PI * x2);
}

/**
 * @brief Writes the analytical solution to a file.
 *
 * Evaluates and saves the exact solution along with the coordinates of each
 * grid point to a file for visualisation and comparison.
 *
 * @param[in] N Number of grid points in each dimension.
 * @param[in] filename Name of the output file.
 */
void write_analytical_solution(int N, const char* filename) {
  double h  = 1.0 / ((double) N + 1);
  FILE*  fp = fopen(filename, "w");
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      double x1    = (i + 1) * h;
      double x2    = (j + 1) * h;
      double exact = analytical_solution(x1, x2);
      fprintf(fp, "%f %f %f\n", x1, x2, exact);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

/**
 * @brief Main function.
 *
 * Runs the CG algorithm for multiple grid sizes, measures performance, and
 * writes both numerical and analytical solutions to files.
 *
 * @return 0 upon successful execution.
 */
int main() {
  int    N_values[] = {8, 16, 32, 64, 128, 256};
  int    num_N      = sizeof(N_values) / sizeof(N_values[0]);
  double tol        = 1e-8;  // Tolerance for convergence
  int    max_iter   = 10000; // Maximum number of iterations
  printf("N\tIterations\tTime (s)\tFinal Residual\n");
  printf("------------------------------------------------------\n");

  // Run the algorithm for each grid size
  for (int i = 0; i < num_N; i++) {
    int N    = N_values[i];
    int size = N * N;

    // Allocate memory
    double* b = (double*) malloc(size * sizeof(double));
    double* x = (double*) malloc(size * sizeof(double));

    // Create right-hand side vector
    rhs(b, N, f);

    // Do a single run first in order to get an iteration count
    int    iter_count;
    double final_residual;
    cg(b, x, N, tol, max_iter, &iter_count, &final_residual);

    // Mulitple runs for accurate timings as per the assignment
    int     num_runs = 5;
    clock_t start    = clock();
    for (int run = 0; run < num_runs; run++) {
      for (int j = 0; j < size; j++) {
        x[j] = 0.0;
      }
      cg(b, x, N, tol, max_iter, &iter_count, &final_residual);
    }
    clock_t end        = clock();
    double  time_spent = (double) (end - start) / (CLOCKS_PER_SEC * num_runs);

    printf("%d\t%d\t\t%.6f\t%.10e\n", N, iter_count, time_spent,
           final_residual);

    // // Add verification against analytical solution
    // verify_solution(x, N);

    // Save the computed solution to a file for plotting
    char filename[50];
    sprintf(filename, "solution-%d.dat", N);
    write_solution(x, N, filename);

    // Write analytical solution for comparison
    char analytical_filename[50];
    sprintf(analytical_filename, "analytical-%d.dat", N);
    write_analytical_solution(N, analytical_filename);

    // Free memory
    free(b);
    free(x);
  }
  return 0;
}
