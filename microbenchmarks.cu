
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <cublas_v2.h>
// #include "microkernels.h"

#define ALIGNMENT 64
#define TRIALS 1000
#define WARMUP_TRIALS 10

#define BLAS


static cublasHandle_t handle;

int main(int argc, char *argv[]) {

    int m = argc > 1 ? atoi(argv[1]) : 768;
    int n = argc > 2 ? atoi(argv[2]) : 2048;

    printf("m: %d, n: %d\n", m, n);
    
    float alpha = 1.0;
    float beta = 0.0;
    float *w, *x, *xout;
    float *w_gpu, *x_gpu, *xout_gpu;

    w = (float*)malloc(m * n * sizeof(float));
    x = (float*)malloc(n * sizeof(float));
    xout = (float*)malloc(m * sizeof(float));

    cudaMalloc((void**)&w_gpu, m * n * sizeof(float));
    cudaMalloc((void**)&x_gpu, n * sizeof(float));
    cudaMalloc((void**)&xout_gpu, m * sizeof(float));



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

    cudaMemcpy(w_gpu, w, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_gpu, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xout_gpu, xout, m * sizeof(float), cudaMemcpyHostToDevice);

    clock_t start, stop;
    double elapsed;
    double total = 0.0;
    double std = 0.0;


    void (*gemv)(float*, float*, float*, int, int);
    // gemv = matmul;



    for (int i = 0; i < WARMUP_TRIALS; i++) {
#ifdef BLAS
        cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, w_gpu, m, x_gpu, 1, &beta, xout_gpu, 1);
#else
        gemv(xout, x, w);
#endif
    }
    for (int i = 0; i < TRIALS; i++) {
        cudaDeviceSynchronize();
        start = clock();
#ifdef BLAS
        cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, w_gpu, m, x_gpu, 1, &beta, xout_gpu, 1);
#else
        gemv(xout, x, w);
#endif
        cudaDeviceSynchronize();
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

    free(x);
    free(w);
    free(xout);
    cublasDestroy(handle);
    return 0;
}