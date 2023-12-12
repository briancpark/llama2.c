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
#include <Accelerate/Accelerate.h>
#define HANDTUNED
// #define BNNS


void matmul_swiglu(float* xout0, float* xout1, float* x, float* w0, float* w1, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val0 = 0.0f;
        float val1 = 0.0f;
        for (int j = 0; j < n; j++) {
            val0 += w0[i * n + j] * x[j];
            val1 += w1[i * n + j] * x[j];
        }
        xout0[i] = val0 * (1.0f / (1.0f + expf(-val0))) * val1;
    }
}


//768
void rmsnorm(float* o, float* x, float* weight, int size) {
#ifdef HANDTUNED
    const int prefetch_distance = 16; // Adjust based on your architecture and workload
    // calculate sum of squares
    register float32x4_t sum_vec = vdupq_n_f32(0.0);  // Initialize a vector of 4 floats to 0

    // Vectorized and unrolled loop
    int j;
    for (j = 0; j <= size - 48; j += 48) {
        __builtin_prefetch(&x[j + prefetch_distance]);
        register float32x4_t x_vec1 = vld1q_f32(&x[j]);
        register float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
        register float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
        register float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
        register float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
        register float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
        register float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
        register float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);
        register float32x4_t x_vec9 = vld1q_f32(&x[j + 32]);
        register float32x4_t x_vec10 = vld1q_f32(&x[j + 36]);
        register float32x4_t x_vec11 = vld1q_f32(&x[j + 40]);
        register float32x4_t x_vec12 = vld1q_f32(&x[j + 44]);

        sum_vec = vmlaq_f32(sum_vec, x_vec1, x_vec1);  // sum_vec += x_vec1 * x_vec1
        sum_vec = vmlaq_f32(sum_vec, x_vec2, x_vec2);
        sum_vec = vmlaq_f32(sum_vec, x_vec3, x_vec3);
        sum_vec = vmlaq_f32(sum_vec, x_vec4, x_vec4);
        sum_vec = vmlaq_f32(sum_vec, x_vec5, x_vec5);
        sum_vec = vmlaq_f32(sum_vec, x_vec6, x_vec6);
        sum_vec = vmlaq_f32(sum_vec, x_vec7, x_vec7);
        sum_vec = vmlaq_f32(sum_vec, x_vec8, x_vec8);
        sum_vec = vmlaq_f32(sum_vec, x_vec9, x_vec9);
        sum_vec = vmlaq_f32(sum_vec, x_vec10, x_vec10);
        sum_vec = vmlaq_f32(sum_vec, x_vec11, x_vec11);
        sum_vec = vmlaq_f32(sum_vec, x_vec12, x_vec12);
    }

    // Horizontal sum of vector
    float32x2_t sum_vec2 = vpadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
    float ss = vget_lane_f32(vpadd_f32(sum_vec2, sum_vec2), 0);

    // Handle remaining elements
    for (; j < size; ++j) {
        ss += x[j] * x[j];
    }

    // Final RMS normalization calculations
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    // Vectorized normalization and scaling with unrolling
    float32x4_t ss_vec = vdupq_n_f32(ss);
    for (j = 0; j <= size - 16; j += 16) {
        float32x4_t x_vec1 = vld1q_f32(&x[j]);
        float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
        float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
        float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);

        float32x4_t weight_vec1 = vld1q_f32(&weight[j]);
        float32x4_t weight_vec2 = vld1q_f32(&weight[j + 4]);
        float32x4_t weight_vec3 = vld1q_f32(&weight[j + 8]);
        float32x4_t weight_vec4 = vld1q_f32(&weight[j + 12]);
        
        float32x4_t result_vec1 = vmulq_f32(vmulq_f32(ss_vec, x_vec1), weight_vec1);
        float32x4_t result_vec2 = vmulq_f32(vmulq_f32(ss_vec, x_vec2), weight_vec2);
        float32x4_t result_vec3 = vmulq_f32(vmulq_f32(ss_vec, x_vec3), weight_vec3);
        float32x4_t result_vec4 = vmulq_f32(vmulq_f32(ss_vec, x_vec4), weight_vec4);

        vst1q_f32(&o[j], result_vec1);
        vst1q_f32(&o[j + 4], result_vec2);
        vst1q_f32(&o[j + 8], result_vec3);
        vst1q_f32(&o[j + 12], result_vec4);
    }
#else
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
#endif
}

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
            register float32x4_t x_vec1 = vld1q_f32(&x[j]);
            register float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            register float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            register float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            register float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            register float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            register float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            register float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);

            __builtin_prefetch(&w[i * 768 + j + prefetch_distance]);
            register float32x4_t w_vec1 = vld1q_f32(&w[i * 768 + j]);
            register float32x4_t w_vec2 = vld1q_f32(&w[i * 768 + j + 4]);
            register float32x4_t w_vec3 = vld1q_f32(&w[i * 768 + j + 8]);
            register float32x4_t w_vec4 = vld1q_f32(&w[i * 768 + j + 12]);
            register float32x4_t w_vec5 = vld1q_f32(&w[i * 768 + j + 16]);
            register float32x4_t w_vec6 = vld1q_f32(&w[i * 768 + j + 20]);
            register float32x4_t w_vec7 = vld1q_f32(&w[i * 768 + j + 24]);
            register float32x4_t w_vec8 = vld1q_f32(&w[i * 768 + j + 28]);

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
            register float32x4_t x_vec1 = vld1q_f32(&x[j]);
            register float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            register float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            register float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            register float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            register float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            register float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            register float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);

            __builtin_prefetch(&w[i * 2048 + j + prefetch_distance]);
            register float32x4_t w_vec1 = vld1q_f32(&w[i * 2048 + j]);
            register float32x4_t w_vec2 = vld1q_f32(&w[i * 2048 + j + 4]);
            register float32x4_t w_vec3 = vld1q_f32(&w[i * 2048 + j + 8]);
            register float32x4_t w_vec4 = vld1q_f32(&w[i * 2048 + j + 12]);
            register float32x4_t w_vec5 = vld1q_f32(&w[i * 2048 + j + 16]);
            register float32x4_t w_vec6 = vld1q_f32(&w[i * 2048 + j + 20]);
            register float32x4_t w_vec7 = vld1q_f32(&w[i * 2048 + j + 24]);
            register float32x4_t w_vec8 = vld1q_f32(&w[i * 2048 + j + 28]);

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
            register float32x4_t x_vec1 = vld1q_f32(&x[j]);
            register float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            register float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            register float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            register float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            register float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            register float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            register float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);

            __builtin_prefetch(&w[i * 768 + j + prefetch_distance]);
            register float32x4_t w_vec1 = vld1q_f32(&w[i * 768 + j]);
            register float32x4_t w_vec2 = vld1q_f32(&w[i * 768 + j + 4]);
            register float32x4_t w_vec3 = vld1q_f32(&w[i * 768 + j + 8]);
            register float32x4_t w_vec4 = vld1q_f32(&w[i * 768 + j + 12]);
            register float32x4_t w_vec5 = vld1q_f32(&w[i * 768 + j + 16]);
            register float32x4_t w_vec6 = vld1q_f32(&w[i * 768 + j + 20]);
            register float32x4_t w_vec7 = vld1q_f32(&w[i * 768 + j + 24]);
            register float32x4_t w_vec8 = vld1q_f32(&w[i * 768 + j + 28]);

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
            register float32x4_t x_vec1 = vld1q_f32(&x[j]);
            register float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            register float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            register float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            register float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            register float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            register float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            register float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);

            __builtin_prefetch(&w[i * 768 + j + prefetch_distance]);
            register float32x4_t w_vec1 = vld1q_f32(&w[i * 768 + j]);
            register float32x4_t w_vec2 = vld1q_f32(&w[i * 768 + j + 4]);
            register float32x4_t w_vec3 = vld1q_f32(&w[i * 768 + j + 8]);
            register float32x4_t w_vec4 = vld1q_f32(&w[i * 768 + j + 12]);
            register float32x4_t w_vec5 = vld1q_f32(&w[i * 768 + j + 16]);
            register float32x4_t w_vec6 = vld1q_f32(&w[i * 768 + j + 20]);
            register float32x4_t w_vec7 = vld1q_f32(&w[i * 768 + j + 24]);
            register float32x4_t w_vec8 = vld1q_f32(&w[i * 768 + j + 28]);

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

bool inited = false;
__fp16* xout_fp16;
__fp16* x_fp16;
__fp16* w_fp16;

void init_buffer() {
    if (inited) {
        return;
    } 
    inited = true;
    xout_fp16 = (__fp16*)malloc(32000 * sizeof(__fp16));
    x_fp16 = (__fp16*)malloc(4096 * sizeof(__fp16));
    w_fp16 = (__fp16*)malloc(4096 * 32000 * sizeof(__fp16));
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
#ifdef BNNS
    init_buffer();
    
    // convert to fp16
    for (int i = 0; i < n; i++) {
        x_fp16[i] = (__fp16)x[i];
    }

    for (int i = 0; i < d * n; i++) {
        w_fp16[i] = (__fp16)w[i];
    }


    BNNSNDArrayDescriptor descA = {
        .flags = 0,
        .layout = BNNSDataLayout2DFirstMajor,
        .size = {d, n},
        .stride = {1, n},
        .data = w_fp16,
        .data_type = BNNSDataTypeFloat16,
    };

    BNNSNDArrayDescriptor descB = {
        .flags = 0,
        .layout = BNNSDataLayout2DFirstMajor,
        .size = {n, 1},
        .stride = {1, 1},
        .data = x_fp16,
        .data_type = BNNSDataTypeFloat16,
    };

    BNNSNDArrayDescriptor descC = {
        .flags = 0,
        .layout = BNNSDataLayout2DFirstMajor,
        .size = {d, 1},
        .stride = {1, 1},
        .data = xout_fp16,
        .data_type = BNNSDataTypeFloat16,
    };

    BNNSFilterParameters filter_params = {
        .n_threads = 1,

    };

    
    int status = BNNSMatMul(false, false, 1.0f, &descA, &descB, &descC, NULL, &filter_params);
    if (status != 0) {
        printf("BNNSMatMul failed with status %d\n", status);
    }

    // convert back to fp32
    for (int i = 0; i < d; i++) {
        xout[i] = (float)xout_fp16[i];
    }

#else
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
        // printf("matmul: xout (%d,), x (%d,), w (%d,%d)\n", d, n, d, n);
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
#endif
}

void matmul_32000_768_add(float* resid, float* xout, float* x, float* w) {
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
            register float32x4_t x_vec1 = vld1q_f32(&x[j]);
            register float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            register float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            register float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            register float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            register float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            register float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            register float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);

            __builtin_prefetch(&w[i * 768 + j + prefetch_distance]);
            register float32x4_t w_vec1 = vld1q_f32(&w[i * 768 + j]);
            register float32x4_t w_vec2 = vld1q_f32(&w[i * 768 + j + 4]);
            register float32x4_t w_vec3 = vld1q_f32(&w[i * 768 + j + 8]);
            register float32x4_t w_vec4 = vld1q_f32(&w[i * 768 + j + 12]);
            register float32x4_t w_vec5 = vld1q_f32(&w[i * 768 + j + 16]);
            register float32x4_t w_vec6 = vld1q_f32(&w[i * 768 + j + 20]);
            register float32x4_t w_vec7 = vld1q_f32(&w[i * 768 + j + 24]);
            register float32x4_t w_vec8 = vld1q_f32(&w[i * 768 + j + 28]);

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

        resid[i] += result[0];
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

void matmul_768_2048_add(float* resid, float* xout, float* x, float* w) {
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
            register float32x4_t x_vec1 = vld1q_f32(&x[j]);
            register float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            register float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            register float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            register float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            register float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            register float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            register float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);

            __builtin_prefetch(&w[i * 2048 + j + prefetch_distance]);
            register float32x4_t w_vec1 = vld1q_f32(&w[i * 2048 + j]);
            register float32x4_t w_vec2 = vld1q_f32(&w[i * 2048 + j + 4]);
            register float32x4_t w_vec3 = vld1q_f32(&w[i * 2048 + j + 8]);
            register float32x4_t w_vec4 = vld1q_f32(&w[i * 2048 + j + 12]);
            register float32x4_t w_vec5 = vld1q_f32(&w[i * 2048 + j + 16]);
            register float32x4_t w_vec6 = vld1q_f32(&w[i * 2048 + j + 20]);
            register float32x4_t w_vec7 = vld1q_f32(&w[i * 2048 + j + 24]);
            register float32x4_t w_vec8 = vld1q_f32(&w[i * 2048 + j + 28]);

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

        resid[i] += result[0];
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

void matmul_768_768_add(float* resid, float* xout, float* x, float* w) {
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
            register float32x4_t x_vec1 = vld1q_f32(&x[j]);
            register float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            register float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            register float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            register float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            register float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            register float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            register float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);

            __builtin_prefetch(&w[i * 768 + j + prefetch_distance]);
            register float32x4_t w_vec1 = vld1q_f32(&w[i * 768 + j]);
            register float32x4_t w_vec2 = vld1q_f32(&w[i * 768 + j + 4]);
            register float32x4_t w_vec3 = vld1q_f32(&w[i * 768 + j + 8]);
            register float32x4_t w_vec4 = vld1q_f32(&w[i * 768 + j + 12]);
            register float32x4_t w_vec5 = vld1q_f32(&w[i * 768 + j + 16]);
            register float32x4_t w_vec6 = vld1q_f32(&w[i * 768 + j + 20]);
            register float32x4_t w_vec7 = vld1q_f32(&w[i * 768 + j + 24]);
            register float32x4_t w_vec8 = vld1q_f32(&w[i * 768 + j + 28]);

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

        resid[i] += result[0];
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

void matmul_2048_768_add(float* resid, float* xout, float* x, float* w) {
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
            register float32x4_t x_vec1 = vld1q_f32(&x[j]);
            register float32x4_t x_vec2 = vld1q_f32(&x[j + 4]);
            register float32x4_t x_vec3 = vld1q_f32(&x[j + 8]);
            register float32x4_t x_vec4 = vld1q_f32(&x[j + 12]);
            register float32x4_t x_vec5 = vld1q_f32(&x[j + 16]);
            register float32x4_t x_vec6 = vld1q_f32(&x[j + 20]);
            register float32x4_t x_vec7 = vld1q_f32(&x[j + 24]);
            register float32x4_t x_vec8 = vld1q_f32(&x[j + 28]);

            __builtin_prefetch(&w[i * 768 + j + prefetch_distance]);
            register float32x4_t w_vec1 = vld1q_f32(&w[i * 768 + j]);
            register float32x4_t w_vec2 = vld1q_f32(&w[i * 768 + j + 4]);
            register float32x4_t w_vec3 = vld1q_f32(&w[i * 768 + j + 8]);
            register float32x4_t w_vec4 = vld1q_f32(&w[i * 768 + j + 12]);
            register float32x4_t w_vec5 = vld1q_f32(&w[i * 768 + j + 16]);
            register float32x4_t w_vec6 = vld1q_f32(&w[i * 768 + j + 20]);
            register float32x4_t w_vec7 = vld1q_f32(&w[i * 768 + j + 24]);
            register float32x4_t w_vec8 = vld1q_f32(&w[i * 768 + j + 28]);

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

        resid[i] += result[0];
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


void matmul_add(float* resid, float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function

    // Stories 110M
    if (n == 768 && d == 32000) {
        matmul_32000_768_add(resid, xout, x, w);
    } else if (n == 2048 && d == 768) {
        matmul_768_2048_add(resid, xout, x, w);
    } else if (n == 768 && d == 768) {
        matmul_768_768_add(resid, xout, x, w);
    } else if (n == 768 && d == 2048) {
        matmul_2048_768_add(resid, xout, x, w);

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
        // printf("matmul: xout (%d,), x (%d,), w (%d,%d)\n", d, n, d, n);
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

// void softmax(float* x, int size) {
//     // find max value (for numerical stability)
//     float max_val = x[0];
//     for (int i = 1; i < size; i++) {
//         if (x[i] > max_val) {
//             max_val = x[i];
//         }
//     }
//     // exp and sum
//     float sum = 0.0f;
//     for (int i = 0; i < size; i++) {
//         x[i] = expf(x[i] - max_val);
//         sum += x[i];
//     }
//     // normalize
//     for (int i = 0; i < size; i++) {
//         x[i] /= sum;
//     }
// }