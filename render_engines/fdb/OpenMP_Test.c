#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define SIZE 1024*8 // Adjust for stress (e.g., 2048 or 4096)

double **alloc_matrix(int size) {
    double **mat = malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
        mat[i] = calloc(size, sizeof(double));
    return mat;
}

void free_matrix(double **mat, int size) {
    for (int i = 0; i < size; i++) free(mat[i]);
    free(mat);
}

int main() {
    int n = SIZE;
    double **A = alloc_matrix(n);
    double **B = alloc_matrix(n);
    double **C = alloc_matrix(n);

    // Initialize matrices A and B
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
        }

    // Start timing
    double start = omp_get_wtime();

    // Matrix multiplication C = A * B
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    double end = omp_get_wtime();
    double time = end - start;

    // FLOP count: 2 * n^3 for dense matmul
    double flops = 2.0 * n * n * n;
    double gflops = flops / (time * 1e9);

    printf("Time: %.3f seconds\n", time);
    printf("Performance: %.2f GFLOPs\n", gflops);

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}

// Compile with: gcc -fopenmp OpenMP_Test.c -o OpenMP_Test && ./OpenMP_Test
// To run with different sizes, change SIZE at the top of the code.
// To compile and run, use the command:





