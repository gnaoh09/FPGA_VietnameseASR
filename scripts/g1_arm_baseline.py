#!/usr/bin/env python3
"""G1 Baseline Benchmark: x86_64 Reference + Local ARM64 (Scalar / NEON Classic / NEON SDOT) & Embedded ARM Scaling."""

import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"

C_SOURCE = r"""
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
"""


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    c_path = BUILD_DIR / "g1_arm_bench.c"
    bin_path = BUILD_DIR / "g1_arm_bench"
    c_path.write_text(C_SOURCE, encoding="utf-8")

    subprocess.run(
        ["clang", "-O3", "-march=armv8.2-a+dotprod", str(c_path), "-o", str(bin_path), "-lm"],
        check=True,
    )
    res = subprocess.run([str(bin_path)], check=True, capture_output=True, text=True)
    raw = json.loads(res.stdout.strip())
    assert raw["mismatch_classic"] == 0, "NEON classic mismatch vs scalar!"
    assert raw["mismatch_sdot"] == 0, "NEON SDOT mismatch vs scalar!"

    macs = raw["shape"]["macs"]
    elems = raw["shape"]["M"] * raw["shape"]["N"]

    kernel_stats = {}
    for k in [
        "gemm_fp32_neon",
        "gemm_int8_scalar",
        "gemm_int8_neon_classic",
        "gemm_int8_neon_sdot",
        "swoosh_fp32",
        "swoosh_int8_lut",
    ]:
        times = raw[k]
        med = statistics.median(times)
        mn = min(times)
        mx = max(times)
        entry = {
            "median_ms": round(med * 1e3, 4),
            "min_ms": round(mn * 1e3, 4),
            "max_ms": round(mx * 1e3, 4),
        }
        if k.startswith("gemm"):
            entry["gmac_per_s_1thread"] = round((macs / med) / 1e9, 3)
        else:
            entry["melem_per_s_1thread"] = round((elems / med) / 1e6, 3)
        kernel_stats[k] = entry

    # Reference x86_64 results from Downloads/README.md (ZipFormer-150M, i7-12700F, num_threads=4)
    # Clean group = test_vivos (686 utt, 0.700h) + test_mcv_vi (1367 utt, 1.446h) = 2053 utt, 2.146h
    x86_datasets = {
        "test_vivos": {"utt": 686, "hours": 0.700, "fp32_wer": 3.82, "int8_wer": 3.88, "fp32_rtf": 0.0191, "int8_rtf": 0.0125, "fp32_rss_mib": 910.93, "int8_rss_mib": 412.30, "fp32_tone_sb": 0.321, "int8_tone_sb": 0.321},
        "test_mcv_vi": {"utt": 1367, "hours": 1.446, "fp32_wer": 5.87, "int8_wer": 6.05, "fp32_rtf": 0.0222, "int8_rtf": 0.0152, "fp32_rss_mib": 902.90, "int8_rss_mib": 451.90, "fp32_tone_sb": 0.682, "int8_tone_sb": 0.703},
        "test_fleurs_vi": {"utt": 841, "hours": 2.871, "fp32_wer": 5.48, "int8_wer": 5.46, "fp32_rtf": 0.0187, "int8_rtf": 0.0120, "fp32_rss_mib": 1290.20, "int8_rss_mib": 758.40, "fp32_tone_sb": 0.115, "int8_tone_sb": 0.103},
        "test_vlsp2020_eval": {"utt": 500, "hours": 1.198, "fp32_wer": 10.83, "int8_wer": 10.77, "fp32_rtf": 0.0203, "int8_rtf": 0.0120, "fp32_rss_mib": 1264.60, "int8_rss_mib": 782.82, "fp32_tone_sb": 0.265, "int8_tone_sb": 0.259},
    }
    clean_fp32_rtf = round((0.700 * 0.0191 + 1.446 * 0.0222) / 2.146, 4)
    clean_int8_rtf = round((0.700 * 0.0125 + 1.446 * 0.0152) / 2.146, 4)

    # Analytical normalization to Embedded ARM targets (T=500 -> 5.0 s audio; 150M=8.4283 GMACs; 30M~1.25 GMACs)
    # Note: 1) Local Apple M5 host (~4.0 GHz P-core) measured directly above.
    #       2) Cortex-A9 Dual @ 800 MHz (no SDOT, 128-bit NEON executing 1x 8-lane vmull/cycle -> ~3.2 GMAC/s peak, ~25-40% sustained for full ZipFormer incl. memory/Swoosh/transpose overhead)
    #       3) Cortex-A53 Quad @ 1.2 GHz (no SDOT, NEON classic)
    # Explicitly report both measured M5 host RTF (1T & scaled) and analytical Cortex-A9/A53 projections.
    enc_macs_150m_g = 8.4283
    enc_macs_30m_g = 1.2500

    # On M5 (1-thread measured), compute-kernel time for 8.43 GMACs:
    m5_fp32_t = enc_macs_150m_g / kernel_stats["gemm_fp32_neon"]["gmac_per_s_1thread"]
    m5_i8_cls_t = enc_macs_150m_g / kernel_stats["gemm_int8_neon_classic"]["gmac_per_s_1thread"]
    m5_i8_sdot_t = enc_macs_150m_g / kernel_stats["gemm_int8_neon_sdot"]["gmac_per_s_1thread"]

    # Embedded Cortex-A9 Dual @ 800 MHz (from Downloads/README.md Sec 16: INT8 150M projected RTF ~0.43-1.08x)
    # With Classic NEON SIMD (vmull_s8+vpadalq_s16) vs Scalar C:
    # Scalar on A9 @ 800 MHz (~0.25 GMAC/s/core * 2 cores * 0.7 = 0.35 GMAC/s): 8.43 / 0.35 = 24.1 s -> RTF ~4.82 (150M), ~0.71 (30M)
    # NEON Classic on A9 Dual @ 800 MHz (~2.5-3.9 GMAC/s sustained incl. Swoosh/Transpose): RTF 0.43-1.08 (150M), 0.08-0.18 (30M)
    a9_scalar_150m_rtf = 4.82
    a9_neon_150m_rtf_range = [0.43, 1.08]
    a9_neon_30m_rtf_range = [0.08, 0.18]
    a53_neon_150m_rtf_range = [0.22, 0.52]
    a53_neon_30m_rtf_range = [0.04, 0.09]

    output = {
        "gate": "G1",
        "host_machine": platform.machine(),
        "host_cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "compiler": subprocess.check_output(["clang", "--version"], text=True).splitlines()[0],
        "correctness": {"neon_classic_vs_scalar_mismatch": raw["mismatch_classic"], "neon_sdot_vs_scalar_mismatch": raw["mismatch_sdot"]},
        "measured_arm64_kernels_1thread": kernel_stats,
        "host_arm64_kernel_only_150m_T500": {
            "fp32_neon_s": round(m5_fp32_t, 4),
            "fp32_neon_rtf": round(m5_fp32_t / 5.0, 4),
            "int8_neon_classic_s": round(m5_i8_cls_t, 4),
            "int8_neon_classic_rtf": round(m5_i8_cls_t / 5.0, 4),
            "int8_neon_sdot_s": round(m5_i8_sdot_t, 4),
            "int8_neon_sdot_rtf": round(m5_i8_sdot_t / 5.0, 4),
            "speedup_sdot_vs_scalar": round(kernel_stats["gemm_int8_scalar"]["median_ms"] / kernel_stats["gemm_int8_neon_sdot"]["median_ms"], 2),
            "speedup_classic_vs_scalar": round(kernel_stats["gemm_int8_scalar"]["median_ms"] / kernel_stats["gemm_int8_neon_classic"]["median_ms"], 2),
            "speedup_swoosh_lut_vs_fp32": round(kernel_stats["swoosh_fp32"]["median_ms"] / kernel_stats["swoosh_int8_lut"]["median_ms"], 2),
        },
        "x86_server_reference_i7_12700F": {
            "datasets": x86_datasets,
            "clean_group_2053utt_fp32_rtf": clean_fp32_rtf,
            "clean_group_2053utt_int8_rtf": clean_int8_rtf,
        },
        "embedded_arm_targets": {
            "cortex_a9_dual_800mhz_no_sdot": {
                "scalar_int8_150m_rtf_est": a9_scalar_150m_rtf,
                "neon_classic_int8_150m_rtf_range": a9_neon_150m_rtf_range,
                "neon_classic_int8_30m_rtf_range": a9_neon_30m_rtf_range,
                "fp32_150m_ram_status": "OOM (~1240 MiB C++ RSS > 1 GB DDR3 on long utterances)",
                "int8_150m_ram_status": "Fits 1 GB DDR3 (~362-733 MiB), OOM on <=512 MB",
            },
            "cortex_a53_quad_1200mhz_no_sdot": {
                "neon_classic_int8_150m_rtf_range": a53_neon_150m_rtf_range,
                "neon_classic_int8_30m_rtf_range": a53_neon_30m_rtf_range,
            },
        },
    }

    out_path = BUILD_DIR / "g1_baseline_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
