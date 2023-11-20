#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "immintrin.h"
#include "mkl.h"

#define HANDTUNED

void matmul_32000_768(float* xout, float* x, float* w) {
#ifdef HANDTUNED
    const int prefetch_distance = 32; // This is a heuristic. Adjust as needed for your specific architecture and workload.

    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < 32000; i++) {
        register __m256 val_vec1, val_vec2, val_vec3, val_vec4;
        register __m256 x_vec1, x_vec2, x_vec3, x_vec4, w_vec1, w_vec2, w_vec3, w_vec4;
        val_vec1 = _mm256_setzero_ps();
        val_vec2 = _mm256_setzero_ps();
        val_vec3 = _mm256_setzero_ps();
        val_vec4 = _mm256_setzero_ps();

        int j;
        for (j = 0; j <= 768 - 32; j += 32) {
            // Prefetching
            _mm_prefetch((const char*) &x[j + prefetch_distance], _MM_HINT_T0);
            _mm_prefetch((const char*) &w[i * 768 + j + prefetch_distance], _MM_HINT_T0);

            x_vec1 = _mm256_load_ps(&x[j]);
            w_vec1 = _mm256_load_ps(&w[i * 768 + j]);
            val_vec1 = _mm256_fmadd_ps(x_vec1, w_vec1, val_vec1);

            x_vec2 = _mm256_load_ps(&x[j + 8]);
            w_vec2 = _mm256_load_ps(&w[i * 768 + j + 8]);
            val_vec2 = _mm256_fmadd_ps(x_vec2, w_vec2, val_vec2);

            x_vec3 = _mm256_load_ps(&x[j + 16]);
            w_vec3 = _mm256_load_ps(&w[i * 768 + j + 16]);          
            val_vec3 = _mm256_fmadd_ps(x_vec3, w_vec3, val_vec3);  

            x_vec4 = _mm256_load_ps(&x[j + 24]);
            w_vec4 = _mm256_load_ps(&w[i * 768 + j + 24]);

            val_vec4 = _mm256_fmadd_ps(x_vec4, w_vec4, val_vec4);
        }

        // Combine all results
        val_vec1 = _mm256_add_ps(val_vec1, val_vec2);
        val_vec3 = _mm256_add_ps(val_vec3, val_vec4);
        val_vec1 = _mm256_add_ps(val_vec1, val_vec3);
    
        // Horizontal sum
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        float result[8];
        _mm256_store_ps(result, val_vec1);
        xout[i] = result[0] + result[4];
    }
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 32000; i++) {
        float val = 0.0f;
        for (int j = 0; j < 768; j++) {
            val += w[i * 768 + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}

void matmul_768_2048(float* xout, float* x, float* w) {
#ifdef HANDTUNED
    const int prefetch_distance = 32; // This is a heuristic. Adjust as needed for your specific architecture and workload.

    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < 768; i++) {
        register __m256 val_vec1, val_vec2, val_vec3, val_vec4;
        register __m256 x_vec1, x_vec2, x_vec3, x_vec4, w_vec1, w_vec2, w_vec3, w_vec4;
        val_vec1 = _mm256_setzero_ps();
        val_vec2 = _mm256_setzero_ps();
        val_vec3 = _mm256_setzero_ps();
        val_vec4 = _mm256_setzero_ps();

        int j;
        for (j = 0; j <= 2048 - 32; j += 32) {
            // Prefetching
            _mm_prefetch((const char*) &x[j + prefetch_distance], _MM_HINT_T0);
            _mm_prefetch((const char*) &w[i * 2048 + j + prefetch_distance], _MM_HINT_T0);

            x_vec1 = _mm256_load_ps(&x[j]);
            w_vec1 = _mm256_load_ps(&w[i * 2048 + j]);
            val_vec1 = _mm256_fmadd_ps(x_vec1, w_vec1, val_vec1);

            x_vec2 = _mm256_load_ps(&x[j + 8]);
            w_vec2 = _mm256_load_ps(&w[i * 2048 + j + 8]);
            val_vec2 = _mm256_fmadd_ps(x_vec2, w_vec2, val_vec2);

            x_vec3 = _mm256_load_ps(&x[j + 16]);
            w_vec3 = _mm256_load_ps(&w[i * 2048 + j + 16]);          
            val_vec3 = _mm256_fmadd_ps(x_vec3, w_vec3, val_vec3);  

            x_vec4 = _mm256_load_ps(&x[j + 24]);
            w_vec4 = _mm256_load_ps(&w[i * 2048 + j + 24]);

            val_vec4 = _mm256_fmadd_ps(x_vec4, w_vec4, val_vec4);
        }

        // Combine all results
        val_vec1 = _mm256_add_ps(val_vec1, val_vec2);
        val_vec3 = _mm256_add_ps(val_vec3, val_vec4);
        val_vec1 = _mm256_add_ps(val_vec1, val_vec3);
    
        // Horizontal sum
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        float result[8];
        _mm256_store_ps(result, val_vec1);
        xout[i] = result[0] + result[4];
    }
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 768; i++) {
        float val = 0.0f;
        for (int j = 0; j < 2048; j++) {
            val += w[i * 2048 + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}

void matmul_768_768(float* xout, float* x, float* w) {
#ifdef HANDTUNED
    const int prefetch_distance = 16; // This is a heuristic. Adjust as needed for your specific architecture and workload.

    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < 768; i++) {
        register __m256 val_vec1, val_vec2, val_vec3, val_vec4;
        register __m256 x_vec1, x_vec2, x_vec3, x_vec4, w_vec1, w_vec2, w_vec3, w_vec4;
        val_vec1 = _mm256_setzero_ps();
        val_vec2 = _mm256_setzero_ps();
        val_vec3 = _mm256_setzero_ps();
        val_vec4 = _mm256_setzero_ps();

        int j;
        for (j = 0; j <= 768 - 32; j += 32) {
            // Prefetching
            _mm_prefetch((const char*) &x[j + prefetch_distance], _MM_HINT_T0);
            _mm_prefetch((const char*) &w[i * 768 + j + prefetch_distance], _MM_HINT_T0);

            x_vec1 = _mm256_load_ps(&x[j]);
            w_vec1 = _mm256_load_ps(&w[i * 768 + j]);
            val_vec1 = _mm256_fmadd_ps(x_vec1, w_vec1, val_vec1);

            x_vec2 = _mm256_load_ps(&x[j + 8]);
            w_vec2 = _mm256_load_ps(&w[i * 768 + j + 8]);
            val_vec2 = _mm256_fmadd_ps(x_vec2, w_vec2, val_vec2);

            x_vec3 = _mm256_load_ps(&x[j + 16]);
            w_vec3 = _mm256_load_ps(&w[i * 768 + j + 16]);          
            val_vec3 = _mm256_fmadd_ps(x_vec3, w_vec3, val_vec3);  

            x_vec4 = _mm256_load_ps(&x[j + 24]);
            w_vec4 = _mm256_load_ps(&w[i * 768 + j + 24]);

            val_vec4 = _mm256_fmadd_ps(x_vec4, w_vec4, val_vec4);
        }

        // Combine all results
        val_vec1 = _mm256_add_ps(val_vec1, val_vec2);
        val_vec3 = _mm256_add_ps(val_vec3, val_vec4);
        val_vec1 = _mm256_add_ps(val_vec1, val_vec3);
    
        // Horizontal sum
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        float result[8];
        _mm256_store_ps(result, val_vec1);
        xout[i] = result[0] + result[4];
    }
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 768; i++) {
        float val = 0.0f;
        for (int j = 0; j < 768; j++) {
            val += w[i * 768 + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}

void matmul_2048_768(float* xout, float* x, float* w) {
#ifdef HANDTUNED
    const int prefetch_distance = 16; // This is a heuristic. Adjust as needed for your specific architecture and workload.

    int i;

// #pragma omp parallel for private(i) num_threads(2)
#pragma omp parallel for private(i)
    for (i = 0; i < 2048; i++) {
        register __m256 val_vec1, val_vec2, val_vec3, val_vec4;
        register __m256 x_vec1, x_vec2, x_vec3, x_vec4, w_vec1, w_vec2, w_vec3, w_vec4;
        val_vec1 = _mm256_setzero_ps();
        val_vec2 = _mm256_setzero_ps();
        val_vec3 = _mm256_setzero_ps();
        val_vec4 = _mm256_setzero_ps();

        int j;
        for (j = 0; j <= 768 - 32; j += 32) {
            // Prefetching
            _mm_prefetch((const char*) &x[j + prefetch_distance], _MM_HINT_T0);
            _mm_prefetch((const char*) &w[i * 768 + j + prefetch_distance], _MM_HINT_T0);

            x_vec1 = _mm256_load_ps(&x[j]);
            w_vec1 = _mm256_load_ps(&w[i * 768 + j]);
            val_vec1 = _mm256_fmadd_ps(x_vec1, w_vec1, val_vec1);

            x_vec2 = _mm256_load_ps(&x[j + 8]);
            w_vec2 = _mm256_load_ps(&w[i * 768 + j + 8]);
            val_vec2 = _mm256_fmadd_ps(x_vec2, w_vec2, val_vec2);

            x_vec3 = _mm256_load_ps(&x[j + 16]);
            w_vec3 = _mm256_load_ps(&w[i * 768 + j + 16]);          
            val_vec3 = _mm256_fmadd_ps(x_vec3, w_vec3, val_vec3);  

            x_vec4 = _mm256_load_ps(&x[j + 24]);
            w_vec4 = _mm256_load_ps(&w[i * 768 + j + 24]);

            val_vec4 = _mm256_fmadd_ps(x_vec4, w_vec4, val_vec4);
        }

        // Combine all results
        val_vec1 = _mm256_add_ps(val_vec1, val_vec2);
        val_vec3 = _mm256_add_ps(val_vec3, val_vec4);
        val_vec1 = _mm256_add_ps(val_vec1, val_vec3);
    
        // Horizontal sum
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        float result[8];
        _mm256_store_ps(result, val_vec1);
        xout[i] = result[0] + result[4];
    }
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 2048; i++) {
        float val = 0.0f;
        for (int j = 0; j < 768; j++) {
            val += w[i * 768 + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}


void matmul_4096_4096(float* xout, float* x, float* w) {
#ifdef HANDTUNED
    const int prefetch_distance = 32; // This is a heuristic. Adjust as needed for your specific architecture and workload.

    int i;

// #pragma omp parallel for private(i)
    for (i = 0; i < 4096; i++) {
        register __m256 val_vec1, val_vec2, val_vec3, val_vec4;
        register __m256 x_vec1, x_vec2, x_vec3, x_vec4, w_vec1, w_vec2, w_vec3, w_vec4;
        val_vec1 = _mm256_setzero_ps();
        val_vec2 = _mm256_setzero_ps();
        val_vec3 = _mm256_setzero_ps();
        val_vec4 = _mm256_setzero_ps();

        int j;
        for (j = 0; j <= 4096 - 32; j += 32) {
            // Prefetching
            _mm_prefetch((const char*) &x[j + prefetch_distance], _MM_HINT_T0);
            _mm_prefetch((const char*) &w[i * 4096 + j + prefetch_distance], _MM_HINT_T0);

            x_vec1 = _mm256_load_ps(&x[j]);
            w_vec1 = _mm256_load_ps(&w[i * 4096 + j]);
            val_vec1 = _mm256_fmadd_ps(x_vec1, w_vec1, val_vec1);

            x_vec2 = _mm256_load_ps(&x[j + 8]);
            w_vec2 = _mm256_load_ps(&w[i * 4096 + j + 8]);
            val_vec2 = _mm256_fmadd_ps(x_vec2, w_vec2, val_vec2);

            x_vec3 = _mm256_load_ps(&x[j + 16]);
            w_vec3 = _mm256_load_ps(&w[i * 4096 + j + 16]);          
            val_vec3 = _mm256_fmadd_ps(x_vec3, w_vec3, val_vec3);  

            x_vec4 = _mm256_load_ps(&x[j + 24]);
            w_vec4 = _mm256_load_ps(&w[i * 4096 + j + 24]);

            val_vec4 = _mm256_fmadd_ps(x_vec4, w_vec4, val_vec4);
        }

        // Combine all results
        val_vec1 = _mm256_add_ps(val_vec1, val_vec2);
        val_vec3 = _mm256_add_ps(val_vec3, val_vec4);
        val_vec1 = _mm256_add_ps(val_vec1, val_vec3);
    
        // Horizontal sum
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        float result[8];
        _mm256_store_ps(result, val_vec1);
        xout[i] = result[0] + result[4];
    }
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 4096; i++) {
        float val = 0.0f;
        for (int j = 0; j < 4096; j++) {
            val += w[i * 4096 + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}


void matmul_11008_4096(float* xout, float* x, float* w) {
#ifdef HANDTUNED
    const int prefetch_distance = 32; // This is a heuristic. Adjust as needed for your specific architecture and workload.

    int i;

#pragma omp parallel for private(i)
    for (i = 0; i < 11008; i++) {
        register __m256 val_vec1, val_vec2, val_vec3, val_vec4;
        register __m256 x_vec1, x_vec2, x_vec3, x_vec4, w_vec1, w_vec2, w_vec3, w_vec4;
        val_vec1 = _mm256_setzero_ps();
        val_vec2 = _mm256_setzero_ps();
        val_vec3 = _mm256_setzero_ps();
        val_vec4 = _mm256_setzero_ps();

        int j;
        for (j = 0; j <= 4096 - 32; j += 32) {
            // Prefetching
            _mm_prefetch((const char*) &x[j + prefetch_distance], _MM_HINT_T0);
            _mm_prefetch((const char*) &w[i * 4096 + j + prefetch_distance], _MM_HINT_T0);

            x_vec1 = _mm256_load_ps(&x[j]);
            w_vec1 = _mm256_load_ps(&w[i * 4096 + j]);
            val_vec1 = _mm256_fmadd_ps(x_vec1, w_vec1, val_vec1);

            x_vec2 = _mm256_load_ps(&x[j + 8]);
            w_vec2 = _mm256_load_ps(&w[i * 4096 + j + 8]);
            val_vec2 = _mm256_fmadd_ps(x_vec2, w_vec2, val_vec2);

            x_vec3 = _mm256_load_ps(&x[j + 16]);
            w_vec3 = _mm256_load_ps(&w[i * 4096 + j + 16]);          
            val_vec3 = _mm256_fmadd_ps(x_vec3, w_vec3, val_vec3);  

            x_vec4 = _mm256_load_ps(&x[j + 24]);
            w_vec4 = _mm256_load_ps(&w[i * 4096 + j + 24]);

            val_vec4 = _mm256_fmadd_ps(x_vec4, w_vec4, val_vec4);
        }

        // Combine all results
        val_vec1 = _mm256_add_ps(val_vec1, val_vec2);
        val_vec3 = _mm256_add_ps(val_vec3, val_vec4);
        val_vec1 = _mm256_add_ps(val_vec1, val_vec3);
    
        // Horizontal sum
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        float result[8];
        _mm256_store_ps(result, val_vec1);
        xout[i] = result[0] + result[4];
    }
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 11008; i++) {
        float val = 0.0f;
        for (int j = 0; j < 4096; j++) {
            val += w[i * 4096 + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}



void matmul_4096_11008(float* xout, float* x, float* w) {
#ifdef HANDTUNED
    const int prefetch_distance = 32; // This is a heuristic. Adjust as needed for your specific architecture and workload.

    int i;

// #pragma omp parallel for private(i)
    for (i = 0; i < 4096; i++) {
        register __m256 val_vec1, val_vec2, val_vec3, val_vec4;
        register __m256 x_vec1, x_vec2, x_vec3, x_vec4, w_vec1, w_vec2, w_vec3, w_vec4;
        val_vec1 = _mm256_setzero_ps();
        val_vec2 = _mm256_setzero_ps();
        val_vec3 = _mm256_setzero_ps();
        val_vec4 = _mm256_setzero_ps();

        int j;
        for (j = 0; j <= 11008 - 32; j += 32) {
            // Prefetching
            _mm_prefetch((const char*) &x[j + prefetch_distance], _MM_HINT_T0);
            _mm_prefetch((const char*) &w[i * 11008 + j + prefetch_distance], _MM_HINT_T0);

            x_vec1 = _mm256_load_ps(&x[j]);
            w_vec1 = _mm256_load_ps(&w[i * 11008 + j]);
            val_vec1 = _mm256_fmadd_ps(x_vec1, w_vec1, val_vec1);

            x_vec2 = _mm256_load_ps(&x[j + 8]);
            w_vec2 = _mm256_load_ps(&w[i * 11008 + j + 8]);
            val_vec2 = _mm256_fmadd_ps(x_vec2, w_vec2, val_vec2);

            x_vec3 = _mm256_load_ps(&x[j + 16]);
            w_vec3 = _mm256_load_ps(&w[i * 11008 + j + 16]);          
            val_vec3 = _mm256_fmadd_ps(x_vec3, w_vec3, val_vec3);  

            x_vec4 = _mm256_load_ps(&x[j + 24]);
            w_vec4 = _mm256_load_ps(&w[i * 11008 + j + 24]);

            val_vec4 = _mm256_fmadd_ps(x_vec4, w_vec4, val_vec4);
        }

        // Combine all results
        val_vec1 = _mm256_add_ps(val_vec1, val_vec2);
        val_vec3 = _mm256_add_ps(val_vec3, val_vec4);
        val_vec1 = _mm256_add_ps(val_vec1, val_vec3);
    
        // Horizontal sum
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        float result[8];
        _mm256_store_ps(result, val_vec1);
        xout[i] = result[0] + result[4];
    }
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 4096; i++) {
        float val = 0.0f;
        for (int j = 0; j < 11008; j++) {
            val += w[i * 11008 + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}

void matmul_32000_4096(float* xout, float* x, float* w) {
#ifdef HANDTUNED
    const int prefetch_distance = 32; // This is a heuristic. Adjust as needed for your specific architecture and workload.

    int i;

#pragma omp parallel for private(i)
    for (i = 0; i < 32000; i++) {
        register __m256 val_vec1, val_vec2, val_vec3, val_vec4;
        register __m256 x_vec1, x_vec2, x_vec3, x_vec4, w_vec1, w_vec2, w_vec3, w_vec4;
        val_vec1 = _mm256_setzero_ps();
        val_vec2 = _mm256_setzero_ps();
        val_vec3 = _mm256_setzero_ps();
        val_vec4 = _mm256_setzero_ps();

        int j;
        for (j = 0; j <= 4096 - 32; j += 32) {
            // Prefetching
            _mm_prefetch((const char*) &x[j + prefetch_distance], _MM_HINT_T0);
            _mm_prefetch((const char*) &w[i * 4096 + j + prefetch_distance], _MM_HINT_T0);

            x_vec1 = _mm256_load_ps(&x[j]);
            w_vec1 = _mm256_load_ps(&w[i * 4096 + j]);
            val_vec1 = _mm256_fmadd_ps(x_vec1, w_vec1, val_vec1);

            x_vec2 = _mm256_load_ps(&x[j + 8]);
            w_vec2 = _mm256_load_ps(&w[i * 4096 + j + 8]);
            val_vec2 = _mm256_fmadd_ps(x_vec2, w_vec2, val_vec2);

            x_vec3 = _mm256_load_ps(&x[j + 16]);
            w_vec3 = _mm256_load_ps(&w[i * 4096 + j + 16]);          
            val_vec3 = _mm256_fmadd_ps(x_vec3, w_vec3, val_vec3);  

            x_vec4 = _mm256_load_ps(&x[j + 24]);
            w_vec4 = _mm256_load_ps(&w[i * 4096 + j + 24]);

            val_vec4 = _mm256_fmadd_ps(x_vec4, w_vec4, val_vec4);
        }

        // Combine all results
        val_vec1 = _mm256_add_ps(val_vec1, val_vec2);
        val_vec3 = _mm256_add_ps(val_vec3, val_vec4);
        val_vec1 = _mm256_add_ps(val_vec1, val_vec3);
    
        // Horizontal sum
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        val_vec1 = _mm256_hadd_ps(val_vec1, val_vec1);
        float result[8];
        _mm256_store_ps(result, val_vec1);
        xout[i] = result[0] + result[4];
    }
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 32000; i++) {
        float val = 0.0f;
        for (int j = 0; j < 4096; j++) {
            val += w[i * 4096 + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}



void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function

    // Stories 110M
    if (n == 768 && d == 32000) {
        matmul_32000_768(xout, x, w);
    } else if (n == 2048 && d == 768) {
        matmul_768_2048(xout, x, w);
    } else if (n == 768 && d == 768) {
        matmul_768_768(xout, x, w);
    } else if (n == 768 && d == 2048) {
        matmul_2048_768(xout, x, w);

    // LLaMA 7B
    } else if (n == 4096 && d == 4096) {
        matmul_4096_4096(xout, x, w);
    } else if (n == 4096 && d == 11008) {
        matmul_11008_4096(xout, x, w);
    } else if (n == 11008 && d == 4096) {
        matmul_4096_11008(xout, x, w);
    } else if (n == 4096 && d == 32000) {
        matmul_32000_4096(xout, x, w);
    } else {     
        // print dimensions
        printf("matmul: xout (%d,), x (%d,), w (%d,%d)\n", d, n, d, n);
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }
}
