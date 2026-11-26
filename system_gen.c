#include <stdlib.h>
#include <math.h>
#include "./system_gen.h"

static double random_uniform(unsigned int *state, double lo, double hi) {
    return lo + (hi - lo) * ((double) rand_r(state) / ((double) RAND_MAX + 1.0));
}

void generate_system(double *A, double *b, int n, unsigned int seed,
                     double margin, int starting, int rowCount) {
  
  for (int i = 0; i < rowCount; i++) {
    int global_i = starting + i;
    unsigned int row_state = seed + (unsigned int) global_i * 2654435761u;
    double row_sum = 0.0;
    
    /* Fill off-diagonal entries of row i */
    for (int j = 0; j < n; j++) {
      if (j == global_i)
        continue; // Skip the diagonal.
      A[i*n + j] = random_uniform(&row_state, 0.0, 1.0); // Row-major indexing of A.
      row_sum += fabs(A[i*n + j]);
    }
    /* Fill the diagonal entry; margin controls the dominance. */
    A[i*n + global_i] = row_sum * (1 + margin);
    b[i] = random_uniform(&row_state, -10.0, 10.0);
  }
}
