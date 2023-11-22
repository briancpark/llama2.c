#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <arm_neon.h>

#define HANDTUNED

void matmul_32000_768(float* xout, float* x, float* w) {
#ifdef HANDTUNED
    const int prefetch_distance = 16; // Adjust based on your architecture and workload

    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < 32000; i++) {
        register float32x4_t val_vec1 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec2 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec3 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec4 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec5 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec6 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec7 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec8 = vdupq_n_f32(0.0f);

        int j;
        for (j = 0; j <= 768 - 32; j += 32) {
            // Prefetching
            __builtin_prefetch(&x[j + prefetch_distance]);
            __builtin_prefetch(&w[i * 768 + j + prefetch_distance]);

            // 1st unroll
            float32x4_t x_vec1 = vld1q_f32(&x[j]);
            float32x4_t w_vec1 = vld1q_f32(&w[i * 768 + j]);
            // val_vec1 = vmlaq_f32(val_vec1, x_vec1, w_vec1);

            // 2nd unroll
            float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            float32x4_t w_vec2 = vld1q_f32(&w[i * 768 + j + 4]);
            // val_vec2 = vmlaq_f32(val_vec2, x_vec2, w_vec2);

            // 3rd unroll
            float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            float32x4_t w_vec3 = vld1q_f32(&w[i * 768 + j + 8]);
            // val_vec3 = vmlaq_f32(val_vec3, x_vec3, w_vec3);

            // 4th unroll
            float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            float32x4_t w_vec4 = vld1q_f32(&w[i * 768 + j + 12]);
            // val_vec4 = vmlaq_f32(val_vec4, x_vec4, w_vec4);

            // 5th unroll
            float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            float32x4_t w_vec5 = vld1q_f32(&w[i * 768 + j + 16]);
            // val_vec5 = vmlaq_f32(val_vec5, x_vec5, w_vec5);

            // 6th unroll
            float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            float32x4_t w_vec6 = vld1q_f32(&w[i * 768 + j + 20]);
            // val_vec6 = vmlaq_f32(val_vec6, x_vec6, w_vec6);

            // 7th unroll
            float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            float32x4_t w_vec7 = vld1q_f32(&w[i * 768 + j + 24]);
            // val_vec7 = vmlaq_f32(val_vec7, x_vec7, w_vec7);

            // 8th unroll
            float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);
            float32x4_t w_vec8 = vld1q_f32(&w[i * 768 + j + 28]);
            val_vec1 = vmlaq_f32(val_vec1, x_vec1, w_vec1);
            val_vec2 = vmlaq_f32(val_vec2, x_vec2, w_vec2);
            val_vec3 = vmlaq_f32(val_vec3, x_vec3, w_vec3);
            val_vec4 = vmlaq_f32(val_vec4, x_vec4, w_vec4);
            val_vec5 = vmlaq_f32(val_vec5, x_vec5, w_vec5);
            val_vec6 = vmlaq_f32(val_vec6, x_vec6, w_vec6);
            val_vec7 = vmlaq_f32(val_vec7, x_vec7, w_vec7);
            val_vec8 = vmlaq_f32(val_vec8, x_vec8, w_vec8);
        }

        // Combine all results
        val_vec1 = vaddq_f32(val_vec1, val_vec2);
        val_vec3 = vaddq_f32(val_vec3, val_vec4);
        val_vec5 = vaddq_f32(val_vec5, val_vec6);
        val_vec7 = vaddq_f32(val_vec7, val_vec8);
        val_vec1 = vaddq_f32(val_vec1, val_vec3);
        val_vec5 = vaddq_f32(val_vec5, val_vec7);
        val_vec1 = vaddq_f32(val_vec1, val_vec5);

        // Horizontal sum
        float32x2_t sum_vec = vadd_f32(vget_high_f32(val_vec1), vget_low_f32(val_vec1)); // Add high and low parts
        sum_vec = vpadd_f32(sum_vec, sum_vec); // Pairwise addition
        float result[2];
        vst1_f32(result, sum_vec);

        xout[i] = result[0];
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
    const int prefetch_distance = 16; // Adjust based on your architecture and workload

    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < 768; i++) {
        register float32x4_t val_vec1 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec2 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec3 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec4 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec5 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec6 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec7 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec8 = vdupq_n_f32(0.0f);

        int j;
        for (j = 0; j <= 2048 - 32; j += 32) {
            // Prefetching
            __builtin_prefetch(&x[j + prefetch_distance]);
            __builtin_prefetch(&w[i * 2048 + j + prefetch_distance]);

            // 1st unroll
            float32x4_t x_vec1 = vld1q_f32(&x[j]);
            float32x4_t w_vec1 = vld1q_f32(&w[i * 2048 + j]);
            // val_vec1 = vmlaq_f32(val_vec1, x_vec1, w_vec1);

            // 2nd unroll
            float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            float32x4_t w_vec2 = vld1q_f32(&w[i * 2048 + j + 4]);
            // val_vec2 = vmlaq_f32(val_vec2, x_vec2, w_vec2);

            // 3rd unroll
            float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            float32x4_t w_vec3 = vld1q_f32(&w[i * 2048 + j + 8]);
            // val_vec3 = vmlaq_f32(val_vec3, x_vec3, w_vec3);

            // 4th unroll
            float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            float32x4_t w_vec4 = vld1q_f32(&w[i * 2048 + j + 12]);
            // val_vec4 = vmlaq_f32(val_vec4, x_vec4, w_vec4);

            // 5th unroll
            float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            float32x4_t w_vec5 = vld1q_f32(&w[i * 2048 + j + 16]);
            // val_vec5 = vmlaq_f32(val_vec5, x_vec5, w_vec5);

            // 6th unroll
            float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            float32x4_t w_vec6 = vld1q_f32(&w[i * 2048 + j + 20]);
            // val_vec6 = vmlaq_f32(val_vec6, x_vec6, w_vec6);

            // 7th unroll
            float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            float32x4_t w_vec7 = vld1q_f32(&w[i * 2048 + j + 24]);
            // val_vec7 = vmlaq_f32(val_vec7, x_vec7, w_vec7);

            // 8th unroll
            float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);
            float32x4_t w_vec8 = vld1q_f32(&w[i * 2048 + j + 28]);
            val_vec1 = vmlaq_f32(val_vec1, x_vec1, w_vec1);
            val_vec2 = vmlaq_f32(val_vec2, x_vec2, w_vec2);
            val_vec3 = vmlaq_f32(val_vec3, x_vec3, w_vec3);
            val_vec4 = vmlaq_f32(val_vec4, x_vec4, w_vec4);
            val_vec5 = vmlaq_f32(val_vec5, x_vec5, w_vec5);
            val_vec6 = vmlaq_f32(val_vec6, x_vec6, w_vec6);
            val_vec7 = vmlaq_f32(val_vec7, x_vec7, w_vec7);
            val_vec8 = vmlaq_f32(val_vec8, x_vec8, w_vec8);
        }

        // Combine all results
        val_vec1 = vaddq_f32(val_vec1, val_vec2);
        val_vec3 = vaddq_f32(val_vec3, val_vec4);
        val_vec5 = vaddq_f32(val_vec5, val_vec6);
        val_vec7 = vaddq_f32(val_vec7, val_vec8);
        val_vec1 = vaddq_f32(val_vec1, val_vec3);
        val_vec5 = vaddq_f32(val_vec5, val_vec7);
        val_vec1 = vaddq_f32(val_vec1, val_vec5);


        // Horizontal sum
        float32x2_t sum_vec = vadd_f32(vget_high_f32(val_vec1), vget_low_f32(val_vec1)); // Add high and low parts
        sum_vec = vpadd_f32(sum_vec, sum_vec); // Pairwise addition
        float result[2];
        vst1_f32(result, sum_vec);

        xout[i] = result[0];
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
    const int prefetch_distance = 16; // Adjust based on your architecture and workload

    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < 768; i++) {
        register float32x4_t val_vec1 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec2 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec3 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec4 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec5 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec6 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec7 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec8 = vdupq_n_f32(0.0f);

        int j;
        for (j = 0; j <= 768 - 32; j += 32) {
            // Prefetching
            __builtin_prefetch(&x[j + prefetch_distance]);
            __builtin_prefetch(&w[i * 768 + j + prefetch_distance]);

            // 1st unroll
            float32x4_t x_vec1 = vld1q_f32(&x[j]);
            float32x4_t w_vec1 = vld1q_f32(&w[i * 768 + j]);
            // val_vec1 = vmlaq_f32(val_vec1, x_vec1, w_vec1);

            // 2nd unroll
            float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            float32x4_t w_vec2 = vld1q_f32(&w[i * 768 + j + 4]);
            // val_vec2 = vmlaq_f32(val_vec2, x_vec2, w_vec2);

            // 3rd unroll
            float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            float32x4_t w_vec3 = vld1q_f32(&w[i * 768 + j + 8]);
            // val_vec3 = vmlaq_f32(val_vec3, x_vec3, w_vec3);

            // 4th unroll
            float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            float32x4_t w_vec4 = vld1q_f32(&w[i * 768 + j + 12]);
            // val_vec4 = vmlaq_f32(val_vec4, x_vec4, w_vec4);

            // 5th unroll
            float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            float32x4_t w_vec5 = vld1q_f32(&w[i * 768 + j + 16]);
            // val_vec5 = vmlaq_f32(val_vec5, x_vec5, w_vec5);

            // 6th unroll
            float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            float32x4_t w_vec6 = vld1q_f32(&w[i * 768 + j + 20]);
            // val_vec6 = vmlaq_f32(val_vec6, x_vec6, w_vec6);

            // 7th unroll
            float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            float32x4_t w_vec7 = vld1q_f32(&w[i * 768 + j + 24]);
            // val_vec7 = vmlaq_f32(val_vec7, x_vec7, w_vec7);

            // 8th unroll
            float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);
            float32x4_t w_vec8 = vld1q_f32(&w[i * 768 + j + 28]);
            val_vec1 = vmlaq_f32(val_vec1, x_vec1, w_vec1);
            val_vec2 = vmlaq_f32(val_vec2, x_vec2, w_vec2);
            val_vec3 = vmlaq_f32(val_vec3, x_vec3, w_vec3);
            val_vec4 = vmlaq_f32(val_vec4, x_vec4, w_vec4);
            val_vec5 = vmlaq_f32(val_vec5, x_vec5, w_vec5);
            val_vec6 = vmlaq_f32(val_vec6, x_vec6, w_vec6);
            val_vec7 = vmlaq_f32(val_vec7, x_vec7, w_vec7);
            val_vec8 = vmlaq_f32(val_vec8, x_vec8, w_vec8);
        }

        // Combine all results
        val_vec1 = vaddq_f32(val_vec1, val_vec2);
        val_vec3 = vaddq_f32(val_vec3, val_vec4);
        val_vec5 = vaddq_f32(val_vec5, val_vec6);
        val_vec7 = vaddq_f32(val_vec7, val_vec8);
        val_vec1 = vaddq_f32(val_vec1, val_vec3);
        val_vec5 = vaddq_f32(val_vec5, val_vec7);
        val_vec1 = vaddq_f32(val_vec1, val_vec5);


        // Horizontal sum
        float32x2_t sum_vec = vadd_f32(vget_high_f32(val_vec1), vget_low_f32(val_vec1)); // Add high and low parts
        sum_vec = vpadd_f32(sum_vec, sum_vec); // Pairwise addition
        float result[2];
        vst1_f32(result, sum_vec);

        xout[i] = result[0];
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
    const int prefetch_distance = 16; // Adjust based on your architecture and workload

    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < 2048; i++) {
        register float32x4_t val_vec1 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec2 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec3 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec4 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec5 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec6 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec7 = vdupq_n_f32(0.0f);
        register float32x4_t val_vec8 = vdupq_n_f32(0.0f);

        int j;
        for (j = 0; j <= 768 - 32; j += 32) {
            // Prefetching
            __builtin_prefetch(&x[j + prefetch_distance]);
            __builtin_prefetch(&w[i * 768 + j + prefetch_distance]);

            // 1st unroll
            float32x4_t x_vec1 = vld1q_f32(&x[j]);
            float32x4_t w_vec1 = vld1q_f32(&w[i * 768 + j]);
            // val_vec1 = vmlaq_f32(val_vec1, x_vec1, w_vec1);

            // 2nd unroll
            float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            float32x4_t w_vec2 = vld1q_f32(&w[i * 768 + j + 4]);
            // val_vec2 = vmlaq_f32(val_vec2, x_vec2, w_vec2);

            // 3rd unroll
            float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            float32x4_t w_vec3 = vld1q_f32(&w[i * 768 + j + 8]);
            // val_vec3 = vmlaq_f32(val_vec3, x_vec3, w_vec3);

            // 4th unroll
            float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            float32x4_t w_vec4 = vld1q_f32(&w[i * 768 + j + 12]);
            // val_vec4 = vmlaq_f32(val_vec4, x_vec4, w_vec4);

            // 5th unroll
            float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            float32x4_t w_vec5 = vld1q_f32(&w[i * 768 + j + 16]);
            // val_vec5 = vmlaq_f32(val_vec5, x_vec5, w_vec5);

            // 6th unroll
            float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            float32x4_t w_vec6 = vld1q_f32(&w[i * 768 + j + 20]);
            // val_vec6 = vmlaq_f32(val_vec6, x_vec6, w_vec6);

            // 7th unroll
            float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            float32x4_t w_vec7 = vld1q_f32(&w[i * 768 + j + 24]);
            // val_vec7 = vmlaq_f32(val_vec7, x_vec7, w_vec7);

            // 8th unroll
            float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);
            float32x4_t w_vec8 = vld1q_f32(&w[i * 768 + j + 28]);
            val_vec1 = vmlaq_f32(val_vec1, x_vec1, w_vec1);
            val_vec2 = vmlaq_f32(val_vec2, x_vec2, w_vec2);
            val_vec3 = vmlaq_f32(val_vec3, x_vec3, w_vec3);
            val_vec4 = vmlaq_f32(val_vec4, x_vec4, w_vec4);
            val_vec5 = vmlaq_f32(val_vec5, x_vec5, w_vec5);
            val_vec6 = vmlaq_f32(val_vec6, x_vec6, w_vec6);
            val_vec7 = vmlaq_f32(val_vec7, x_vec7, w_vec7);
            val_vec8 = vmlaq_f32(val_vec8, x_vec8, w_vec8);
        }

        // Combine all results
        val_vec1 = vaddq_f32(val_vec1, val_vec2);
        val_vec3 = vaddq_f32(val_vec3, val_vec4);
        val_vec5 = vaddq_f32(val_vec5, val_vec6);
        val_vec7 = vaddq_f32(val_vec7, val_vec8);
        val_vec1 = vaddq_f32(val_vec1, val_vec3);
        val_vec5 = vaddq_f32(val_vec5, val_vec7);
        val_vec1 = vaddq_f32(val_vec1, val_vec5);


        // Horizontal sum
        float32x2_t sum_vec = vadd_f32(vget_high_f32(val_vec1), vget_low_f32(val_vec1)); // Add high and low parts
        sum_vec = vpadd_f32(sum_vec, sum_vec); // Pairwise addition
        float result[2];
        vst1_f32(result, sum_vec);

        xout[i] = result[0];
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
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 4096; i++) {
        float val = 0.0f;
        for (int j = 0; j < 4096; j++) {
            val += w[i * 4096 + j] * x[j];
        }
        xout[i] = val;
    }
}


void matmul_11008_4096(float* xout, float* x, float* w) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 11008; i++) {
        float val = 0.0f;
        for (int j = 0; j < 4096; j++) {
            val += w[i * 4096 + j] * x[j];
        }
        xout[i] = val;
    }
}



void matmul_4096_11008(float* xout, float* x, float* w) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 4096; i++) {
        float val = 0.0f;
        for (int j = 0; j < 11008; j++) {
            val += w[i * 11008 + j] * x[j];
        }
        xout[i] = val;
    }
}

void matmul_32000_4096(float* xout, float* x, float* w) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < 32000; i++) {
        float val = 0.0f;
        for (int j = 0; j < 4096; j++) {
            val += w[i * 4096 + j] * x[j];
        }
        xout[i] = val;
    }
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
