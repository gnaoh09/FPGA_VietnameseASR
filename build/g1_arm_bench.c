
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void gemm_fp32(const float *A, const float *B_T, float *C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        const float *a_row = A + m * K;
        float *c_row = C + m * N;
        for (int n = 0; n < N; ++n) {
            const float *b_row = B_T + n * K;
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(a_row + k), vld1q_f32(b_row + k));
                acc1 = vfmaq_f32(acc1, vld1q_f32(a_row + k + 4), vld1q_f32(b_row + k + 4));
            }
            float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
            for (; k < K; ++k) sum += a_row[k] * b_row[k];
            c_row[n] = sum;
        }
    }
}

__attribute__((noinline))
static void gemm_int8_scalar(const int8_t *A, const int8_t *B_T, int32_t *C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        const int8_t *a_row = A + m * K;
        int32_t *c_row = C + m * N;
        for (int n = 0; n < N; ++n) {
            const int8_t *b_row = B_T + n * K;
            int32_t acc = 0;
            #pragma clang loop vectorize(disable) unroll(disable)
            for (int k = 0; k < K; ++k) {
                acc += (int32_t)a_row[k] * (int32_t)b_row[k];
            }
            c_row[n] = acc;
        }
    }
}

/* Classic ARMv7/ARMv8 NEON without SDOT (matches Cortex-A9 / Cortex-A53 NEON capability) */
static void gemm_int8_neon_classic(const int8_t *A, const int8_t *B_T, int32_t *C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        const int8_t *a_row = A + m * K;
        int32_t *c_row = C + m * N;
        for (int n = 0; n < N; ++n) {
            const int8_t *b_row = B_T + n * K;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                int8x16_t va = vld1q_s8(a_row + k);
                int8x16_t vb = vld1q_s8(b_row + k);
                int16x8_t p_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
                int16x8_t p_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
                acc0 = vpadalq_s16(acc0, p_lo);
                acc1 = vpadalq_s16(acc1, p_hi);
            }
            int32_t sum = vaddvq_s32(vaddq_s32(acc0, acc1));
            for (; k < K; ++k) sum += (int32_t)a_row[k] * (int32_t)b_row[k];
            c_row[n] = sum;
        }
    }
}

/* ARMv8.2+ NEON SDOT (available on modern ARM cores, absent on Cortex-A9/A53) */
static void gemm_int8_neon_sdot(const int8_t *A, const int8_t *B_T, int32_t *C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        const int8_t *a_row = A + m * K;
        int32_t *c_row = C + m * N;
        for (int n = 0; n < N; ++n) {
            const int8_t *b_row = B_T + n * K;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                acc0 = vdotq_s32(acc0, vld1q_s8(a_row + k), vld1q_s8(b_row + k));
                acc1 = vdotq_s32(acc1, vld1q_s8(a_row + k + 16), vld1q_s8(b_row + k + 16));
            }
            int32_t sum = vaddvq_s32(vaddq_s32(acc0, acc1));
            for (; k < K; ++k) sum += (int32_t)a_row[k] * (int32_t)b_row[k];
            c_row[n] = sum;
        }
    }
}

static void swoosh_fp32(const float *in, float *out, int count) {
    for (int i = 0; i < count; ++i) {
        float x = in[i];
        /* SwooshL: log(1 + exp(x - 1)) - 0.08 * x - 0.313261687 */
        float z = x - 1.0f;
        float softplus = (z > 20.0f) ? z : log1pf(expf(z));
        out[i] = softplus - 0.08f * x - 0.313261687f;
    }
}

static void swoosh_int8_lut(const int8_t *in, int8_t *out, const int8_t *lut256, int count) {
    for (int i = 0; i < count; ++i) {
        out[i] = lut256[(uint8_t)in[i]];
    }
}

int main(void) {
    const int M = 128;
    const int K = 512;
    const int N = 1536;
    const int RUNS = 10;
    const double macs = (double)M * (double)N * (double)K;

    float *A_f = (float *)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float *B_f = (float *)aligned_alloc(64, (size_t)N * K * sizeof(float));
    float *C_f = (float *)aligned_alloc(64, (size_t)M * N * sizeof(float));
    int8_t *A_i8 = (int8_t *)aligned_alloc(64, (size_t)M * K);
    int8_t *B_i8 = (int8_t *)aligned_alloc(64, (size_t)N * K);
    int32_t *C_ref = (int32_t *)aligned_alloc(64, (size_t)M * N * sizeof(int32_t));
    int32_t *C_cls = (int32_t *)aligned_alloc(64, (size_t)M * N * sizeof(int32_t));
    int32_t *C_sdot = (int32_t *)aligned_alloc(64, (size_t)M * N * sizeof(int32_t));
    int8_t *lut_in = (int8_t *)aligned_alloc(64, (size_t)M * N);
    int8_t *lut_out = (int8_t *)aligned_alloc(64, (size_t)M * N);
    int8_t lut256[256];

    for (int i = 0; i < M * K; ++i) {
        A_i8[i] = (int8_t)(((i * 37 + 13) % 127) - 63);
        A_f[i] = (float)A_i8[i] * 0.05f;
    }
    for (int i = 0; i < N * K; ++i) {
        B_i8[i] = (int8_t)(((i * 53 + 7) % 127) - 63);
        B_f[i] = (float)B_i8[i] * 0.05f;
    }
    for (int i = 0; i < 256; ++i) {
        int8_t x = (int8_t)i;
        float xf = (float)x * 0.1f;
        float y = log1pf(expf(xf - 1.0f)) - 0.08f * xf - 0.313261687f;
        int q = (int)lrintf(y / 0.1f);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        lut256[i] = (int8_t)q;
    }
    for (int i = 0; i < M * N; ++i) lut_in[i] = (int8_t)(i & 0xFF);

    /* Correctness check */
    gemm_int8_scalar(A_i8, B_i8, C_ref, M, N, K);
    gemm_int8_neon_classic(A_i8, B_i8, C_cls, M, N, K);
    gemm_int8_neon_sdot(A_i8, B_i8, C_sdot, M, N, K);
    int mismatch_cls = memcmp(C_ref, C_cls, (size_t)M * N * sizeof(int32_t)) != 0;
    int mismatch_sdot = memcmp(C_ref, C_sdot, (size_t)M * N * sizeof(int32_t)) != 0;

    /* Warmup */
    gemm_fp32(A_f, B_f, C_f, M, N, K);
    swoosh_fp32(C_f, A_f, M * K);
    volatile int64_t sink = 0;

    printf("{");
    printf("\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d,\"macs\":%.0f},", M, N, K, macs);
    printf("\"mismatch_classic\":%d,\"mismatch_sdot\":%d,", mismatch_cls, mismatch_sdot);

    #define BENCH(NAME, REPS, CALL, ACC_EXPR) do { \
        printf("\"%s\":[", NAME); \
        for (int r = 0; r < RUNS; ++r) { \
            double t0 = now_sec(); \
            for (int rep = 0; rep < (REPS); ++rep) { \
                CALL; \
                sink += (int64_t)(ACC_EXPR); \
            } \
            double t1 = now_sec(); \
            printf("%.9f%s", (t1 - t0) / (double)(REPS), (r + 1 < RUNS) ? "," : ""); \
        } \
        printf("]"); \
    } while (0)

    BENCH("gemm_fp32_neon", 1, gemm_fp32(A_f, B_f, C_f, M, N, K), C_f[0]); printf(",");
    BENCH("gemm_int8_scalar", 1, gemm_int8_scalar(A_i8, B_i8, C_ref, M, N, K), C_ref[0]); printf(",");
    BENCH("gemm_int8_neon_classic", 1, gemm_int8_neon_classic(A_i8, B_i8, C_cls, M, N, K), C_cls[0]); printf(",");
    BENCH("gemm_int8_neon_sdot", 1, gemm_int8_neon_sdot(A_i8, B_i8, C_sdot, M, N, K), C_sdot[0]); printf(",");
    BENCH("swoosh_fp32", 20, swoosh_fp32(B_f, C_f, M * N), C_f[0]); printf(",");
    BENCH("swoosh_int8_lut", 200, swoosh_int8_lut(lut_in, lut_out, lut256, M * N), lut_out[(rep * 17) % (M * N)]);
    printf(",\"sink\":%lld}\n", (long long)sink);
    return 0;
}
