
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <immintrin.h>
#include <mkl.h>
#include "microkernels.h"

#define ALIGNMENT 64
#define TRIALS 1000
#define WARMUP_TRIALS 10

#define BLAS

int main(int argc, char *argv[]) {

    int m = argc > 1 ? atoi(argv[1]) : 768;
    int n = argc > 2 ? atoi(argv[2]) : 2048;

    printf("m: %d, n: %d\n", m, n);
    
    float *w, *x, *xout;

    w = (float*)mkl_malloc(m * n * sizeof(float), ALIGNMENT);
    x = (float*)mkl_malloc(n * sizeof(float), ALIGNMENT);
    xout = (float*)mkl_malloc(m * sizeof(float), ALIGNMENT);

    if (w == NULL || x == NULL || xout == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // set xout to 0
    memset(xout, 0, m * sizeof(float));
    
    
    for (int i = 0; i < m * n; i++) {
        w[i] = rand();
    }

    for (int i = 0; i < n; i++) {
        x[i] = rand();
    }


    clock_t start, stop;
    double elapsed;
    double total = 0.0;
    double std = 0.0;

    // assigne gemv to be a function pointer to the microkernel based on the input size

    // first declare the function pointer
    void (*gemv)(float*, float*, float*);
    if (m == 32000 && n == 768)
        gemv = matmul_32000_768;
    else if (m == 768 && n == 2048)
        gemv = matmul_768_2048;
    else if (m == 768 && n == 768)
        gemv = matmul_768_768;
    else if (m == 2048 && n == 768)
        gemv = matmul_2048_768;
    else if (m == 4096 && n == 4096)
        gemv = matmul_4096_4096;
    else if (m == 11008 && n == 4096)
        gemv = matmul_11008_4096;
    else if (m == 4096 && n == 11008)
        gemv = matmul_4096_11008;
    else if (m == 32000 && n == 4096)
        gemv = matmul_32000_4096;
    else {
        printf("Invalid input size\n");
        return 1;
    }



    for (int i = 0; i < WARMUP_TRIALS; i++) {
#ifdef BLAS
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, w, n, x, 1, 0.0, xout, 1);
#else
        gemv(xout, x, w);
#endif
    }
    for (int i = 0; i < TRIALS; i++) {
        start = clock();
#ifdef BLAS
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, w, n, x, 1, 0.0, xout, 1);
#else
        gemv(xout, x, w);
#endif
        stop = clock();
        elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
        total += elapsed;
        std += elapsed * elapsed;

        // reset xout
        memset(xout, 0, m * sizeof(float));
        // sleep for 0.01 seconds
        usleep(1000);
    }
    
    double avg = total / TRIALS;
    double var = std / TRIALS - avg * avg;
    double stddev = sqrt(var);
    printf("matmul_768_2048: %f\n", avg);
    printf("matmul_768_2048: %f\n", stddev);
    // print the flops for sgemv
    printf("FLOPS: %f\n", (2.0 * m * n) / (total / TRIALS));
    // print GFLOPS
    printf("GFLOPS: %f\n", (2.0 * m * n) / (total / TRIALS) / 1e9);
    // print TFLOPS
    printf("TFLOPS: %f\n", (2.0 * m * n) / (total / TRIALS) / 1e12);

    mkl_free(x);
    mkl_free(w);
    mkl_free(xout);
    return 0;
}