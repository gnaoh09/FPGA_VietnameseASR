#!/usr/bin/env python3
"""Controlled W4_g64 vs W3_g64 quantization ablation + 6x6 Vietnamese tone
confusion matrix analysis on VIVOS Test (760 utts, 19 speakers, num_threads=1).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import sys
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import sherpa_onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_30m_wer_pareto_noise import (
    BOOT_B,
    BOOT_SEED,
    BUILD_DIR,
    MODEL_DIR,
    QUANT_DIR,
    TONE_COMBINING_MARKS,
    build_quantized_encoder,
    load_vivos_test,
    norm_vi,
    paired_bootstrap_delta,
    quantize_weight_groupwise_symmetric,
    strip_tones_only,
)

TONE_MARK_TO_IDX = {
    "\u0300": 1,  # huyen
    "\u0301": 2,  # sac
    "\u0309": 3,  # hoi
    "\u0303": 4,  # nga
    "\u0323": 5,  # nang
}
TONE_NAMES = ["ngang", "huyen", "sac", "hoi", "nga", "nang"]

EXPECTED_WER = {
    "30M_FP32": 4.610,
    "30M_fpga_bitexact_w8a8": 4.584,
    "30M_fpga_bitexact_w4a8": 4.688,
    "30M_ffn_w3_g64": 4.882,
}


def decompose_vi_syllable(syl: str) -> tuple[str, int]:
    """Return (base_preserving_vowel_diacritics_and_d, tone_idx 0..5)."""
    nfd = unicodedata.normalize("NFD", syl.upper())
    tone_idx = 0
    kept = []
    for ch in nfd:
        if ch in TONE_COMBINING_MARKS:
            tone_idx = TONE_MARK_TO_IDX.get(ch, 0)
        else:
            kept.append(ch)
    return unicodedata.normalize("NFC", "".join(kept)), tone_idx


def align_word_pairs(ref_words: list[str], hyp_words: list[str]) -> list[tuple[str, str]]:
    """Standard Levenshtein word DP backtrace returning 1-to-1 aligned (ref, hyp) pairs."""
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ri = ref_words[i - 1]
        dpi = dp[i]
        dpp = dp[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ri == hyp_words[j - 1] else 1
            sub = dpp[j - 1] + cost
            dele = dpp[j] + 1
            ins = dpi[j - 1] + 1
            dpi[j] = sub if (sub <= dele and sub <= ins) else (dele if dele <= ins else ins)

    pairs: list[tuple[str, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                pairs.append((ref_words[i - 1], hyp_words[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _decode_one_model(args: tuple[str, str, str, str]) -> dict:
    label, enc_path, dec_path, jn_path = args
    ds = load_vivos_test()
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=str(MODEL_DIR / "config.json"),
        encoder=enc_path,
        decoder=dec_path,
        joiner=jn_path,
        num_threads=1,
    )
    n = len(ds)
    wer_err = np.zeros(n, dtype=np.int32)
    wer_ref = np.zeros(n, dtype=np.int32)
    notone_err = np.zeros(n, dtype=np.int32)
    notone_ref = np.zeros(n, dtype=np.int32)
    speakers: list[str] = []
    utterances: list[dict] = []

    for idx, (uid, ref_norm, audio, sr) in enumerate(ds):
        stream = rec.create_stream()
        stream.accept_waveform(sr, audio)
        rec.decode_stream(stream)
        hyp = norm_vi(stream.result.text)

        wo = jiwer.process_words(ref_norm, hyp)
        wer_err[idx] = wo.substitutions + wo.deletions + wo.insertions
        wer_ref[idx] = wo.hits + wo.substitutions + wo.deletions

        nto = jiwer.process_words(strip_tones_only(ref_norm), strip_tones_only(hyp))
        notone_err[idx] = nto.substitutions + nto.deletions + nto.insertions
        notone_ref[idx] = nto.hits + nto.substitutions + nto.deletions

        spk = uid.split("_")[0]
        speakers.append(spk)
        utterances.append({"utt_id": uid, "speaker": spk, "ref": ref_norm, "hyp": hyp})

    wer_pct = round(100.0 * float(wer_err.sum()) / float(wer_ref.sum()), 3)
    notone_pct = round(100.0 * float(notone_err.sum()) / float(notone_ref.sum()), 3)
    return {
        "label": label,
        "wer_pct": wer_pct,
        "wer_no_tone_pct": notone_pct,
        "tone_only_gap_pct": round(wer_pct - notone_pct, 3),
        "stats": {
            "wer_err": wer_err,
            "wer_ref": wer_ref,
            "notone_err": notone_err,
            "notone_ref": notone_ref,
            "clusters": speakers,
        },
        "utterances": utterances,
    }


def compute_tone6x6_per_utt(utterances: list[dict]) -> dict:
    """Compute 6x6 same-base Vietnamese tone confusion matrix & per-utt arrays."""
    n = len(utterances)
    total_ref_by_tone = np.zeros(6, dtype=np.int64)
    same_base_mat_utt = np.zeros((n, 6, 6), dtype=np.int64)
    speakers = [u["speaker"] for u in utterances]

    for ui, u in enumerate(utterances):
        r_words = u["ref"].split()
        h_words = u["hyp"].split()
        for rw in r_words:
            _, rt = decompose_vi_syllable(rw)
            total_ref_by_tone[rt] += 1
        for rw, hw in align_word_pairs(r_words, h_words):
            rb, rt = decompose_vi_syllable(rw)
            hb, ht = decompose_vi_syllable(hw)
            if rb and rb == hb:
                same_base_mat_utt[ui, rt, ht] += 1

    mat = same_base_mat_utt.sum(axis=0)
    tot_ref = int(total_ref_by_tone.sum())
    tot_same_base = int(mat.sum())
    tot_sub_err = int(tot_same_base - np.trace(mat))

    per_tone = {}
    for t, tname in enumerate(TONE_NAMES):
        denom = int(mat[t].sum())
        err = int(denom - mat[t, t])
        per_tone[tname] = {
            "total_ref_syllables": int(total_ref_by_tone[t]),
            "aligned_same_base_denom": denom,
            "tone_errors": err,
            "error_rate_pct": round(100.0 * err / denom, 4) if denom > 0 else 0.0,
        }

    return {
        "tone_order": TONE_NAMES,
        "total_ref_syllables": tot_ref,
        "aligned_same_base_syllables": tot_same_base,
        "same_base_coverage_pct": round(100.0 * tot_same_base / tot_ref, 2),
        "overall_same_base_tone_errors": tot_sub_err,
        "overall_same_base_tone_err_pct": round(100.0 * tot_sub_err / tot_same_base, 4),
        "per_tone": per_tone,
        "confusion_matrix_6x6": mat.tolist(),
        "_utt_mats": same_base_mat_utt,
        "_speakers": speakers,
    }


def speaker_bootstrap_tone_delta(base_t: dict, cand_t: dict) -> dict:
    """19-speaker clustered paired bootstrap (B=2000) for same-base tone error deltas."""
    rng = np.random.default_rng(BOOT_SEED)
    b_mats = base_t["_utt_mats"]
    c_mats = cand_t["_utt_mats"]
    speakers = base_t["_speakers"]

    uniq: dict[str, list[int]] = {}
    for i, spk in enumerate(speakers):
        uniq.setdefault(spk, []).append(i)
    grp_lists = list(uniq.values())
    g = len(grp_lists)
    assert g == 19

    # Group [G, 6, 6] counts
    gb = np.array([b_mats[ix].sum(axis=0) for ix in grp_lists], dtype=np.float64)
    gc = np.array([c_mats[ix].sum(axis=0) for ix in grp_lists], dtype=np.float64)

    gb_den = gb.sum(axis=2)  # [G, 6]
    gc_den = gc.sum(axis=2)
    gb_err = gb_den - np.diagonal(gb, axis1=1, axis2=2)  # [G, 6]
    gc_err = gc_den - np.diagonal(gc, axis1=1, axis2=2)

    gb_tot_den = gb_den.sum(axis=1)
    gc_tot_den = gc_den.sum(axis=1)
    gb_tot_err = gb_err.sum(axis=1)
    gc_tot_err = gc_err.sum(axis=1)

    d_tot = np.empty(BOOT_B, dtype=np.float64)
    d_tone = np.empty((BOOT_B, 6), dtype=np.float64)
    for b in range(BOOT_B):
        draw = rng.integers(0, g, size=g)
        b_rate = 100.0 * gb_tot_err[draw].sum() / gb_tot_den[draw].sum()
        c_rate = 100.0 * gc_tot_err[draw].sum() / gc_tot_den[draw].sum()
        d_tot[b] = c_rate - b_rate

        br = 100.0 * gb_err[draw].sum(axis=0) / np.maximum(gb_den[draw].sum(axis=0), 1.0)
        cr = 100.0 * gc_err[draw].sum(axis=0) / np.maximum(gc_den[draw].sum(axis=0), 1.0)
        d_tone[b] = cr - br

    pt_tot = round(
        float(cand_t["overall_same_base_tone_err_pct"] - base_t["overall_same_base_tone_err_pct"]), 4
    )
    per_tone_delta = {}
    for t, tname in enumerate(TONE_NAMES):
        pt_t = cand_t["per_tone"][tname]["error_rate_pct"] - base_t["per_tone"][tname]["error_rate_pct"]
        per_tone_delta[tname] = {
            "delta_err_pct": round(float(pt_t), 4),
            "ci95": [round(float(np.percentile(d_tone[:, t], 2.5)), 4), round(float(np.percentile(d_tone[:, t], 97.5)), 4)],
        }

    return {
        "n_speakers": g,
        "overall_delta_err_pct": pt_tot,
        "overall_ci95": [round(float(np.percentile(d_tot, 2.5)), 4), round(float(np.percentile(d_tot, 97.5)), 4)],
        "per_tone_delta": per_tone_delta,
    }


def main() -> int:
    enc_fp32 = MODEL_DIR / "encoder-epoch-20-avg-10.onnx"
    dec_fp32 = str(MODEL_DIR / "decoder-epoch-20-avg-10.onnx")
    jn_fp32 = str(MODEL_DIR / "joiner-epoch-20-avg-10.onnx")

    enc_w4_g64 = QUANT_DIR / "encoder_ffn_w4_g64.onnx"
    w4_g64_meta = build_quantized_encoder(
        enc_fp32,
        enc_w4_g64,
        scope="ffn",
        quant_fn=lambda w: quantize_weight_groupwise_symmetric(w, 4, 64),
        dynamic_a8=False,
    )

    jobs = [
        ("30M_FP32", str(enc_fp32), dec_fp32, jn_fp32),
        ("30M_fpga_bitexact_w8a8", str(QUANT_DIR / "encoder_fpga_bitexact_w8a8.onnx"), dec_fp32, jn_fp32),
        ("30M_fpga_bitexact_w4a8", str(QUANT_DIR / "encoder_fpga_bitexact_w4a8.onnx"), dec_fp32, jn_fp32),
        ("30M_ffn_w4_g64", str(enc_w4_g64), dec_fp32, jn_fp32),
        ("30M_ffn_w3_g64", str(QUANT_DIR / "encoder_ffn_w3_g64.onnx"), dec_fp32, jn_fp32),
    ]

    decoded: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for res in pool.map(_decode_one_model, jobs):
            lbl = res["label"]
            if lbl in EXPECTED_WER:
                assert abs(res["wer_pct"] - EXPECTED_WER[lbl]) < 1e-6, (
                    f"{lbl} mismatch: got {res['wer_pct']} vs {EXPECTED_WER[lbl]}"
                )
            decoded[lbl] = res

    s_fp = decoded["30M_FP32"]["stats"]
    s_w4g = decoded["30M_ffn_w4_g64"]["stats"]
    s_w3g = decoded["30M_ffn_w3_g64"]["stats"]

    controlled_ablation = {
        "w4_g64_meta": w4_g64_meta,
        "w4_g64_wer_pct": decoded["30M_ffn_w4_g64"]["wer_pct"],
        "w3_g64_wer_pct": decoded["30M_ffn_w3_g64"]["wer_pct"],
        "w4_g64_vs_fp32_utt": paired_bootstrap_delta(s_fp, s_w4g, cluster_grouped=False),
        "w4_g64_vs_fp32_spk": paired_bootstrap_delta(s_fp, s_w4g, cluster_grouped=True),
        "w3_g64_vs_fp32_utt": paired_bootstrap_delta(s_fp, s_w3g, cluster_grouped=False),
        "w3_g64_vs_fp32_spk": paired_bootstrap_delta(s_fp, s_w3g, cluster_grouped=True),
        "w3_g64_minus_w4_g64_utt": paired_bootstrap_delta(s_w4g, s_w3g, cluster_grouped=False),
        "w3_g64_minus_w4_g64_spk": paired_bootstrap_delta(s_w4g, s_w3g, cluster_grouped=True),
    }

    tone_raw = {lbl: compute_tone6x6_per_utt(decoded[lbl]["utterances"]) for lbl in decoded}
    tone_report = {}
    for lbl, tr in tone_raw.items():
        tone_report[lbl] = {k: v for k, v in tr.items() if not k.startswith("_")}

    tone_deltas_spk = {
        "w4a8_bitexact_minus_fp32": speaker_bootstrap_tone_delta(tone_raw["30M_FP32"], tone_raw["30M_fpga_bitexact_w4a8"]),
        "w8a8_bitexact_minus_fp32": speaker_bootstrap_tone_delta(tone_raw["30M_FP32"], tone_raw["30M_fpga_bitexact_w8a8"]),
        "w3_g64_minus_w4_g64": speaker_bootstrap_tone_delta(tone_raw["30M_ffn_w4_g64"], tone_raw["30M_ffn_w3_g64"]),
    }

    out = {
        "config": {"dataset": "vivos_test", "num_utts": 760, "num_speakers": 19, "num_threads": 1, "boot_b": BOOT_B},
        "controlled_w4_vs_w3_g64_ablation": controlled_ablation,
        "tone_6x6_matrices": tone_report,
        "tone_6x6_speaker_bootstrap_deltas": tone_deltas_spk,
    }
    out_path = BUILD_DIR / "w4_w3_ablation_and_tone6x6.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
