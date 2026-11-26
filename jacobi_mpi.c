#include "./system_gen.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
  Computes a 2-norm of a vector v.
*/
static double vector_norm(double* v, int n) {

    double sum_square = 0.0;
    for (int i = 0; i < n; i++)
        sum_square += v[i] * v[i];

    return sqrt(sum_square);
}

/*
  Computes a 2-norm of Ax-b for a local segment.
*/
static double residual_norm(double* A, double* x, double* b, int local_n, int n) {

    double sum_square = 0.0;

    for (int i = 0; i < local_n; i++) {
        double Ax_i = 0.0;

        for (int j = 0; j < n; j++)
            Ax_i += A[i * n + j] * x[j];

        double r = Ax_i - b[i];
        sum_square += r * r;
    }

    return sum_square;
}

/*
  Solves Ax=b via the Jacobi method with an initial solution in x.

  Inputs
    Max_iter: maximum number of Jacobi iterations to try.
    tolerance: relative residual error used to declare convergence.
    x: the initial solution vector of size n.

  Output:
    x: the solution to Ax=b.
*/

static void jacobi(double* A, double* b, double* x, int n, int local_n,
    int max_iter, double tolerance, int *row_counts, int *first,
    int offset, int rank) {

    double* x_new = malloc((size_t)(local_n > 0 ? local_n : 1) * sizeof(double));
    double local_b_square = 0.0;
    double global_b_square = 0.0;

    for (int i = 0; i < local_n; i++)
        local_b_square += b[i] * b[i];

    MPI_Allreduce(&local_b_square, &global_b_square, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double b_norm = sqrt(global_b_square);

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < local_n; i++) {
            int global_i = offset + i;

            double sigma = 0.0;
            for (int j = 0; j < n; j++)
                if (j != global_i)
                    sigma += A[i * n + j] * x[j];
            x_new[i] = (b[i] - sigma) / A[i * n + global_i];
        }

        MPI_Allgatherv(x_new, local_n, MPI_DOUBLE, x, row_counts, first, MPI_DOUBLE, MPI_COMM_WORLD);

        // Gather all the portions of x_new into x to be checked for tolerance

        // If the solution converged, stop iterating.
        // Global residual norm so all processes terminate after same iteration

        double global_accuracy = 0.0;
        double local_accuracy = residual_norm(A, x, b, local_n, n);
        MPI_Allreduce(&local_accuracy, &global_accuracy, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);

        double sqrt_global_accuracy = sqrt(global_accuracy);

        if (sqrt_global_accuracy / b_norm < tolerance) {
            if (rank == 0) {
                printf("Solved with error %e in %d iterations. \n", sqrt_global_accuracy / b_norm, iter);
            }
            break;
        }
    }

    free(x_new);
    if (iter == max_iter && rank == 0)
        printf("Solver did not converge in %d iterations. \n", max_iter);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Please specify n the size of the square linear system "
                "and margin.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    double margin = atof(argv[2]);

    // splitting up rows
    int* row_counts = malloc(size * sizeof(int));
    int* first =
        malloc(size * sizeof(int)); // first is first element for each scatter row

    for (int current_row = 0, current_distributed = 0; current_row < size;
        ++current_row) {
        row_counts[current_row] = n / size;

        if (current_row < n % size)
            row_counts[current_row]++; // gets an extra row because uneven

        first[current_row] = current_distributed; // assigns first index to be used
        // for that distributed process

        current_distributed = current_distributed + row_counts[current_row]; // go to the next unused
    }

    int local_n = row_counts[rank];
    int offset = first[rank];

    double* x = calloc(n, sizeof(double));
    double* local_A =
        malloc((size_t)(local_n > 0 ? local_n : 1) * n * sizeof(double));
    double* local_b =
        malloc((size_t)(local_n > 0 ? local_n : 1) * sizeof(double));

    // checking valididity

    int safe = x && local_A && local_b;
    // all exist
    if (!safe) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Blocks
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    unsigned int seed = 12345;

    generate_system(local_A, local_b, n, seed,  margin, offset, local_n);


    double tolerance = 1e-10;
    int max_iter = 500;

    // calling jacobi
    jacobi(local_A, local_b, x, n, local_n, max_iter, tolerance, row_counts,first, offset, rank);

    // waiting barrier
    MPI_Barrier(MPI_COMM_WORLD);

    // calculate timing
    double runtime = MPI_Wtime() - start_time;
    if (rank == 0) {
        printf("Runtime: %f seconds\n", runtime);
    }

    // freedom    
    free(local_A);
    free(local_b);
    free(x);
    free(row_counts);
    free(first);

    MPI_Finalize();
    return 0;
}
