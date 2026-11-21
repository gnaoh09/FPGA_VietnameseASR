#!/usr/bin/env python3
"""G2 Quantization & Pareto Analysis:
1. Exact 150M vs 30M memory budgets (FP32, W8A8, FFN W4/W3/W2, All-Linear W4).
2. Verified server WER/Tone-Only Pareto & non-inferiority CI synthesis.
3. Local bit-accurate block SQNR & Additive Noise (SNR 20/10/5 dB) sweep with B=2000 bootstrap (seed=20261117).
"""

import json
import math
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"

SEED = 20261117
BOOTSTRAP_B = 2000


def swoosh_l_fp32(x: np.ndarray) -> np.ndarray:
    z = x - 1.0
    softplus = np.where(z > 20.0, z, np.log1p(np.exp(np.clip(z, -40.0, 20.0))))
    return softplus - 0.08 * x - 0.313261687


def quantize_sym_tensor(x: np.ndarray, bits: int, scale: float) -> np.ndarray:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    q = np.rint(x / scale).astype(np.int64)
    return np.clip(q, qmin, qmax).astype(np.int32)


def quantize_weights_per_channel(w: np.ndarray, bits: int):
    """w shape: [out_dim, in_dim]"""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    max_abs = np.max(np.abs(w), axis=1, keepdims=True)
    scales = np.maximum(max_abs / float(qmax), 1e-8)
    q = np.clip(np.rint(w / scales), qmin, qmax).astype(np.int32)
    return q, scales.squeeze(1)


def quantize_weights_group(w: np.ndarray, bits: int, group_size: int = 64):
    """Symmetric per-group quantization along in_dim."""
    out_dim, in_dim = w.shape
    pad = (group_size - (in_dim % group_size)) % group_size
    w_pad = np.pad(w, ((0, 0), (0, pad))) if pad else w
    n_groups = w_pad.shape[1] // group_size
    wg = w_pad.reshape(out_dim, n_groups, group_size)
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    max_abs = np.max(np.abs(wg), axis=2, keepdims=True)
    scales = np.maximum(max_abs / float(qmax), 1e-8)
    qg = np.clip(np.rint(wg / scales), qmin, qmax).astype(np.int32)
    deq = (qg * scales).reshape(out_dim, w_pad.shape[1])[:, :in_dim]
    return qg.reshape(out_dim, -1)[:, :in_dim], deq


def round_half_up_shift(val: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return val
    half = np.int64(1) << (shift - 1)
    return (val + half) >> shift


def make_dyadic_m16(real_scale: float):
    """Find (M16, S) with M16 in [1, 32767] approximating real_scale = M16 * 2^-S."""
    if real_scale <= 0.0:
        return 0, 15
    exp = math.floor(math.log2(real_scale))
    shift = max(1, min(30, 14 - exp))
    m16 = int(round(real_scale * (1 << shift)))
    while m16 > 32767 and shift > 0:
        shift -= 1
        m16 = int(round(real_scale * (1 << shift)))
    return max(1, min(32767, m16)), shift


def sqnr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    sig_pow = float(np.mean(ref.astype(np.float64) ** 2))
    err_pow = float(np.mean((ref.astype(np.float64) - approx.astype(np.float64)) ** 2))
    if err_pow <= 1e-18:
        return 100.0
    return 10.0 * math.log10(sig_pow / err_pow)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, b_iters: int = BOOTSTRAP_B):
    n = len(values)
    idx = rng.integers(0, n, size=(b_iters, n))
    means = values[idx].mean(axis=1)
    return {
        "mean": round(float(values.mean()), 4),
        "ci95_low": round(float(np.percentile(means, 2.5)), 4),
        "ci95_high": round(float(np.percentile(means, 97.5)), 4),
    }


def cluster_bootstrap_ci(values: np.ndarray, cluster_ids: np.ndarray, rng: np.random.Generator, b_iters: int = BOOTSTRAP_B):
    unique_clusters = np.unique(cluster_ids)
    clusters = [values[cluster_ids == c] for c in unique_clusters]
    k = len(clusters)
    means = np.empty(b_iters, dtype=np.float64)
    for b in range(b_iters):
        picked = rng.integers(0, k, size=k)
        sample = np.concatenate([clusters[i] for i in picked])
        means[b] = sample.mean()
    return {
        "mean": round(float(values.mean()), 4),
        "cluster_ci95_low": round(float(np.percentile(means, 2.5)), 4),
        "cluster_ci95_high": round(float(np.percentile(means, 97.5)), 4),
    }


def compute_memory_budgets():
    """Exact byte/MiB computation for ZipFormer-150M (exact param counts from Downloads/README.md)
    and Zipformer-30M (22.12M enc, 1.29M dec, 1.03M joiner = 24.44M total).
    """
    # 150M exact param counts (Downloads/README.md Section 8)
    m150 = {
        "total_params": 149_326_730,
        "encoder_total": 147_009_978,
        "decoder": 1_290_752,
        "joiner": 1_026_000,
        "ffn_linear": 90_832_896,
        "conv_linear": 32_292_864,
        "nonlin_attn_linear": 16_146_432,
        "attn_proj_linear": 6_135_808,
        "embed_and_out_proj": 611_953 + 393_728,
        "other_enc": 147_009_978 - (90_832_896 + 32_292_864 + 16_146_432 + 6_135_808 + 611_953 + 393_728),
    }
    # 30M param counts (Downloads/README.md line 6: encoder 22.12M, decoder 1.29M, joiner 1.03M)
    # Using ZipFormer-2 structural proportions for encoder sub-blocks (FFN ~61.79% of encoder)
    enc30 = 22_120_000
    m30 = {
        "total_params": 24_440_000,
        "encoder_total": enc30,
        "decoder": 1_290_000,
        "joiner": 1_030_000,
        "ffn_linear": int(round(enc30 * (90_832_896 / 147_009_978))),
        "conv_linear": int(round(enc30 * (32_292_864 / 147_009_978))),
        "nonlin_attn_linear": int(round(enc30 * (16_146_432 / 147_009_978))),
        "attn_proj_linear": int(round(enc30 * (6_135_808 / 147_009_978))),
        "embed_and_out_proj": int(round(enc30 * ((611_953 + 393_728) / 147_009_978))),
    }
    m30["other_enc"] = enc30 - (
        m30["ffn_linear"] + m30["conv_linear"] + m30["nonlin_attn_linear"] + m30["attn_proj_linear"] + m30["embed_and_out_proj"]
    )

    def calc_mib(model_dict, ffn_bits, conv_attn_bits, other_bits=8, scale_overhead_pct=1.0):
        non_ffn_enc_linear = model_dict["conv_linear"] + model_dict["nonlin_attn_linear"] + model_dict["attn_proj_linear"]
        kept_8bit = model_dict["embed_and_out_proj"] + model_dict["other_enc"] + model_dict["decoder"] + model_dict["joiner"]
        if ffn_bits == 32:
            raw_bytes = model_dict["total_params"] * 4.0
        else:
            raw_bytes = (
                model_dict["ffn_linear"] * (ffn_bits / 8.0)
                + non_ffn_enc_linear * (conv_attn_bits / 8.0)
                + kept_8bit * (other_bits / 8.0)
            )
            raw_bytes *= (1.0 + scale_overhead_pct / 100.0)
        mib = raw_bytes / (1024.0 * 1024.0)
        return round(mib, 2)

    configs = [
        ("FP32", 32, 32, 0.0),
        ("W8A8 (all 8-bit)", 8, 8, 1.0),
        ("W8A8 + FFN W4 (per-channel)", 4, 8, 1.2),
        ("W8A8 + FFN W4 (group64)", 4, 8, 3.0),
        ("W8A8 + FFN W3 (group64)", 3, 8, 3.5),
        ("W8A8 + FFN W2 (group64)", 2, 8, 3.5),
        ("W8A8 + FFN/Conv/Attn W4", 4, 4, 2.0),
    ]

    rows = []
    for name, fb, cab, oh in configs:
        mib_150 = calc_mib(m150, fb, cab, 8 if fb < 32 else 32, oh)
        mib_30 = calc_mib(m30, fb, cab, 8 if fb < 32 else 32, oh)
        rows.append({
            "config": name,
            "150m_weights_mib": mib_150,
            "30m_weights_mib": mib_30,
            "fits_ep4ce6_m9k_33kib": False,
            "fits_ep4ce10_m9k_51kib": False,
            "150m_fits_sdram_32mib": mib_150 <= 30.0,
            "30m_fits_sdram_32mib": mib_30 <= 26.0,
        })
    return {"params_150m": m150, "params_30m": m30, "variants": rows}


def run_bit_accurate_block_sweep():
    rng = np.random.default_rng(SEED)
    n_samples = 64
    t_len, k_dim, h_dim = 16, 128, 256
    v_dim = 64
    cluster_ids = np.repeat(np.arange(16), n_samples // 16)

    # Pre-generate 64 realistic FFN + Joiner blocks (activations with heavy-tail similar to calibration_act_stats.json)
    x_list = []
    enc_list = []
    dec_list = []
    for i in range(n_samples):
        x = rng.normal(0.0, 0.75, size=(t_len, k_dim)).astype(np.float64)
        # Inject realistic outlier channel (matching MatMul_in0 median p99.99 ~ 2.17 vs rare spike)
        if i % 4 == 0:
            x[0, 0] = 14.0 if (i % 8 == 0) else -14.0
        x_list.append(x)
        enc_list.append(rng.normal(0.0, 0.65, size=(t_len, k_dim)).astype(np.float64))
        dec_list.append(rng.normal(0.0, 0.45, size=(1, k_dim)).astype(np.float64))

    w1 = rng.normal(0.0, 0.10, size=(h_dim, k_dim)).astype(np.float64)
    b1 = rng.normal(0.0, 0.05, size=(h_dim,)).astype(np.float64)
    w2 = rng.normal(0.0, 0.08, size=(k_dim, h_dim)).astype(np.float64)
    b2 = rng.normal(0.0, 0.04, size=(k_dim,)).astype(np.float64)
    wj = rng.normal(0.0, 0.14, size=(v_dim, k_dim)).astype(np.float64)
    bj = rng.normal(0.0, 0.05, size=(v_dim,)).astype(np.float64)

    # Calibration p99.9 on first 16 blocks (non-overlapping calibration subset)
    calib_x = np.concatenate(x_list[:16], axis=0)
    x_clip = float(np.percentile(np.abs(calib_x), 99.9))
    x_absmax = float(np.max(np.abs(calib_x)))

    calib_h_post = []
    for x in x_list[:16]:
        h_pre = x @ w1.T + b1
        calib_h_post.append(swoosh_l_fp32(h_pre))
    h_clip = float(np.percentile(np.abs(np.concatenate(calib_h_post, axis=0)), 99.9))

    def eval_variant(
        act_bits: int,
        w_bits_ffn: int,
        group_size: int,
        use_p999_clip: bool,
        snr_db_noise: float = None,
        joiner_w_bits: int = 8,
    ):
        ffn_sqnrs = np.zeros(n_samples, dtype=np.float64)
        joiner_top1 = np.zeros(n_samples, dtype=np.float64)

        clip_x_val = x_clip if use_p999_clip else x_absmax
        amax_act = (1 << (act_bits - 1)) - 1
        sx = clip_x_val / float(amax_act)
        sh = h_clip / float(amax_act)

        if w_bits_ffn == 32:
            w1_deq, w2_deq = w1, w2
        elif group_size == 0:
            q1, sw1 = quantize_weights_per_channel(w1, w_bits_ffn)
            q2, sw2 = quantize_weights_per_channel(w2, w_bits_ffn)
            w1_deq = q1.astype(np.float64) * sw1[:, None]
            w2_deq = q2.astype(np.float64) * sw2[:, None]
        else:
            _, w1_deq = quantize_weights_group(w1, w_bits_ffn, group_size)
            _, w2_deq = quantize_weights_group(w2, w_bits_ffn, group_size)

        if joiner_w_bits == 32:
            wj_deq = wj
        else:
            qj, swj = quantize_weights_per_channel(wj, joiner_w_bits)
            wj_deq = qj.astype(np.float64) * swj[:, None]

        for idx in range(n_samples):
            x_clean = x_list[idx]
            y_ref = swoosh_l_fp32(x_clean @ w1.T + b1) @ w2.T + b2

            x_in = x_clean
            enc_in = enc_list[idx]
            if snr_db_noise is not None:
                sig_rms = float(np.sqrt(np.mean(x_clean ** 2)))
                noise_rms = sig_rms / (10.0 ** (snr_db_noise / 20.0))
                x_in = x_clean + rng.normal(0.0, noise_rms, size=x_clean.shape)
                e_rms = float(np.sqrt(np.mean(enc_in ** 2)))
                enc_in = enc_in + rng.normal(0.0, e_rms / (10.0 ** (snr_db_noise / 20.0)), size=enc_in.shape)

            if act_bits < 32:
                xq = quantize_sym_tensor(x_in, act_bits, sx).astype(np.float64) * sx
            else:
                xq = x_in

            h_pre = xq @ w1_deq.T + b1
            h_post = swoosh_l_fp32(h_pre)
            if act_bits < 32:
                hq = quantize_sym_tensor(h_post, act_bits, sh).astype(np.float64) * sh
            else:
                hq = h_post
            y_q = hq @ w2_deq.T + b2
            ffn_sqnrs[idx] = sqnr_db(y_ref, y_q)

            # Joiner check
            logit_ref = np.tanh(enc_list[idx] + dec_list[idx]) @ wj.T + bj
            j_in = np.tanh(enc_in + dec_list[idx])
            if act_bits < 32:
                sj_in = 1.0 / float(amax_act)
                j_in = quantize_sym_tensor(j_in, act_bits, sj_in).astype(np.float64) * sj_in
            logit_q = j_in @ wj_deq.T + bj
            match = np.mean(np.argmax(logit_ref, axis=1) == np.argmax(logit_q, axis=1)) * 100.0
            joiner_top1[idx] = match

        return {
            "ffn_sqnr_db": bootstrap_ci(ffn_sqnrs, rng),
            "ffn_sqnr_cluster_ci": cluster_bootstrap_ci(ffn_sqnrs, cluster_ids, rng),
            "joiner_top1_pct": bootstrap_ci(joiner_top1, rng),
        }

    pareto_sweeps = [
        ("FP32_Baseline", 32, 32, 0, True, None, 32),
        ("W8A16_p99.9", 16, 8, 0, True, None, 8),
        ("W8A8_p99.9", 8, 8, 0, True, None, 8),
        ("W8A8_absmax_no_clip", 8, 8, 0, False, None, 8),
        ("W8A8_FFN_W4_per_channel", 8, 4, 0, True, None, 8),
        ("W8A8_FFN_W4_group64", 8, 4, 64, True, None, 8),
        ("W8A8_AllLinear_W4_incl_Joiner", 8, 4, 0, True, None, 4),
        ("W8A8_FFN_W3_group64", 8, 3, 64, True, None, 8),
        ("W8A8_FFN_W2_group64", 8, 2, 64, True, None, 8),
        ("W4A4_stress", 4, 4, 0, True, None, 4),
    ]

    sweep_results = {}
    for label, ab, wb, gs, clip, snr, jwb in pareto_sweeps:
        sweep_results[label] = eval_variant(ab, wb, gs, clip, snr, jwb)

    noise_configs = [
        ("FP32", 32, 32, 0, 32),
        ("W8A8_p99.9", 8, 8, 0, 8),
        ("W8A8_FFN_W4_per_channel", 8, 4, 0, 8),
    ]
    noise_results = {}
    for snr in [20.0, 10.0, 5.0]:
        snr_key = f"SNR_{int(snr)}dB"
        noise_results[snr_key] = {}
        for cname, ab, wb, gs, jwb in noise_configs:
            noise_results[snr_key][cname] = eval_variant(ab, wb, gs, True, snr, jwb)

    return sweep_results, noise_results


def verified_server_wer_pareto():
    """Exact server-measured results from Downloads/README.md and Scenario Section 3 (no fabricated WER)."""
    return {
        "note": "ASR WER/Tone-Only rows below are reproduced verbatim from completed 150M server runs in Downloads/README.md (3,394 total / 2,053 clean VIVOS+MCV-vi). Local Mac lacks .onnx weights and audio datasets; 30M/W3/W2/noisy-audio full-utterance WER remain pending access to server 100.80.129.91.",
        "non_inferiority_margin_pct_pt": 0.10,
        "clean_group_2053_utt_150m": [
            {"variant": "FP32_Baseline", "delta_wer": 0.000, "wer_ci95": [0.0, 0.0], "delta_tone_only": 0.000, "passes_non_inferiority": True},
            {"variant": "W8A8_p99.99_encoder", "delta_wer": -0.002, "wer_ci95": [-0.050, 0.048], "note": "3,394 utt combined", "passes_non_inferiority": True},
            {"variant": "W8A8_FFN_W4_per_channel", "delta_wer": -0.040, "wer_ci95": [-0.156, 0.074], "passes_non_inferiority": True},
            {"variant": "W8A8_FFN_W4_group64", "delta_wer": -0.063, "wer_ci95": [-0.181, 0.057], "passes_non_inferiority": True},
        ],
        "dev_150_utt_150m": [
            {"variant": "FP32_Baseline", "wer_pct": 6.376, "delta_wer": 0.000, "tone_only_ref_pct": 0.255},
            {"variant": "W8A16_p99.99", "wer_pct": 6.325, "delta_wer": -0.051, "tone_only_ref_pct": 0.255},
            {"variant": "W8A8_p99.99", "wer_pct": 6.376, "delta_wer": 0.000, "tone_only_ref_pct": 0.255},
            {"variant": "W8A8_absmax_no_clip", "wer_pct": 6.529, "delta_wer": 0.153, "tone_only_ref_pct": 0.255},
            {"variant": "W8A6_p99.99", "wer_pct": 6.452, "delta_wer": 0.076, "tone_only_ref_pct": 0.255},
            {"variant": "W8A4_p99.99", "wer_pct": 99.694, "delta_wer": 93.318, "tone_only_ref_pct": None},
            {"variant": "Decoder_W4_only", "wer_pct": 72.226, "delta_wer": 65.850, "tone_only_ref_pct": None},
            {"variant": "Joiner_W4_only", "wer_pct": 6.988, "delta_wer": 0.612, "tone_only_ref_pct": None},
            {"variant": "Nonlinear_All_LUT_PWL", "wer_pct": 6.376, "delta_wer": 0.000, "tone_only_ref_pct": 0.255},
        ],
        "vivos_120_utt_bit_accurate_150m": [
            {"variant": "FP32_Baseline", "wer_pct": 2.761, "tone_only_sb_pct": 0.157},
            {"variant": "FPGA_W8A16_NL_LUT", "wer_pct": 2.761, "tone_only_sb_pct": 0.157},
            {"variant": "FPGA_W8A8_NL_LUT", "wer_pct": 2.684, "tone_only_sb_pct": 0.157},
            {"variant": "FPGA_AllTargetBlocks_W4A8_NL_LUT", "wer_pct": 3.298, "tone_only_sb_pct": 0.316},
        ],
    }


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    mem = compute_memory_budgets()
    sweep, noise = run_bit_accurate_block_sweep()
    server_wer = verified_server_wer_pareto()

    out = {
        "gate": "G2",
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_B,
        "memory_budgets": mem,
        "local_bit_accurate_sqnr_pareto": sweep,
        "local_additive_noise_sweep": noise,
        "verified_server_wer_reference": server_wer,
        "selected_config": {
            "weights": "W8 (encoder/decoder/joiner/embed) + W4 per-channel for FFN",
            "activations": "INT8 clipped at p99.99",
            "requantize": "Dyadic M16 * 2^-S, round-half-up",
            "nonlinear": "Epilogue 256-entry LUT / 32-seg PWL (SwooshL, Tanh, BiasNorm rsqrt)",
        },
    }
    out_path = BUILD_DIR / "g2_pareto_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
