#!/usr/bin/env python3
"""Paired Bootstrap (B=2000) comparing bit-exact FFN W4A8 vs W8A8 and
evaluating both utterance-level (N=760) and speaker-clustered (G=19)
confidence intervals across all main clean VIVOS 30M variants at num_threads=1.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_30m_wer_pareto_noise import (
    BUILD_DIR,
    MODEL_DIR,
    QUANT_DIR,
    eval_recognizer,
    load_vivos_test,
    paired_bootstrap_delta,
)

EXPECTED_VIVOS_WER_PCT = {
    "30M_FP32": 4.610,
    "30M_fpga_bitexact_w8a8": 4.584,
    "30M_fpga_bitexact_w4a8": 4.688,
}


def pair_outcomes(base_stats: dict, cand_stats: dict) -> dict:
    """Count per-utterance better/tied/worse error counts between candidate and baseline."""
    diff = cand_stats["wer_err"] - base_stats["wer_err"]
    return {
        "cand_better_utts": int(np.sum(diff < 0)),
        "tied_utts": int(np.sum(diff == 0)),
        "cand_worse_utts": int(np.sum(diff > 0)),
    }


def with_speaker_clusters(stats: dict) -> dict:
    """Return a shallow copy of stats with speaker IDs extracted from utterance IDs."""
    speakers = [cid.split("_")[0] for cid in stats["clusters"]]
    uniq = set(speakers)
    assert len(uniq) == 19, f"Expected 19 VIVOS Test speakers, got {len(uniq)}"
    return {**stats, "clusters": speakers}


def _run_one(args: tuple[str, str, str, str]) -> tuple[str, dict, dict]:
    label, enc, dec, jn = args
    ds = load_vivos_test()
    s, st = eval_recognizer(
        "vivos", label, Path(enc), Path(dec), Path(jn), ds, num_threads=1
    )
    return label, s, st


def main() -> int:
    dec_fp32 = str(MODEL_DIR / "decoder-epoch-20-avg-10.onnx")
    jn_fp32 = str(MODEL_DIR / "joiner-epoch-20-avg-10.onnx")
    dec_int8 = str(MODEL_DIR / "decoder-epoch-20-avg-10.int8.onnx")
    jn_int8 = str(MODEL_DIR / "joiner-epoch-20-avg-10.int8.onnx")

    jobs = [
        ("30M_FP32", str(MODEL_DIR / "encoder-epoch-20-avg-10.onnx"), dec_fp32, jn_fp32),
        ("30M_fpga_bitexact_w8a8", str(QUANT_DIR / "encoder_fpga_bitexact_w8a8.onnx"), dec_fp32, jn_fp32),
        ("30M_W8A8", str(MODEL_DIR / "encoder-epoch-20-avg-10.int8.onnx"), dec_int8, jn_int8),
        ("30M_fpga_bitexact_w4a8", str(QUANT_DIR / "encoder_fpga_bitexact_w4a8.onnx"), dec_fp32, jn_fp32),
        ("30M_ffn_w3_g64", str(QUANT_DIR / "encoder_ffn_w3_g64.onnx"), dec_fp32, jn_fp32),
        ("30M_all_w4_perch", str(QUANT_DIR / "encoder_all_w4_perch.onnx"), dec_fp32, jn_fp32),
        ("30M_ffn_w2_u4_g64", str(QUANT_DIR / "encoder_ffn_w2_u4_g64.onnx"), dec_fp32, jn_fp32),
    ]

    results: dict[str, tuple[dict, dict]] = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for label, s, st in pool.map(_run_one, jobs):
            if label in EXPECTED_VIVOS_WER_PCT:
                expected = EXPECTED_VIVOS_WER_PCT[label]
                got = round(float(s["wer_pct"]), 3)
                assert abs(got - expected) < 1e-6, (
                    f"{label} VIVOS WER mismatch at num_threads=1: got {got:.3f}%, expected {expected:.3f}%"
                )
            results[label] = (s, st)

    v_fp_s, v_fp = results["30M_FP32"]
    v_w8_s, v_w8 = results["30M_fpga_bitexact_w8a8"]
    v_w4_s, v_w4 = results["30M_fpga_bitexact_w4a8"]

    spk_stats = {k: with_speaker_clusters(st) for k, (_s, st) in results.items()}
    v_fp_spk = spk_stats["30M_FP32"]
    v_w8_spk = spk_stats["30M_fpga_bitexact_w8a8"]
    v_w4_spk = spk_stats["30M_fpga_bitexact_w4a8"]

    all_models_report = {}
    for label, (summ, st) in results.items():
        if label == "30M_FP32":
            all_models_report[label] = {
                "wer_pct": summ["wer_pct"],
                "tone_only_gap_pct": summ["tone_only_gap_pct"],
            }
            continue
        all_models_report[label] = {
            "wer_pct": summ["wer_pct"],
            "tone_only_gap_pct": summ["tone_only_gap_pct"],
            "utterance_bootstrap_vs_fp32": {
                **paired_bootstrap_delta(v_fp, st, cluster_grouped=False),
                **pair_outcomes(v_fp, st),
            },
            "speaker_clustered_bootstrap_vs_fp32": paired_bootstrap_delta(
                v_fp_spk, spk_stats[label], cluster_grouped=True
            ),
        }

    out = {
        "config": {
            "dataset": "vivos_test",
            "num_utts": 760,
            "num_speakers": 19,
            "num_threads": 1,
            "boot_b": 2000,
        },
        "vivos_clean": {
            "fp32_wer_pct": v_fp_s["wer_pct"],
            "w8a8_wer_pct": v_w8_s["wer_pct"],
            "w4a8_wer_pct": v_w4_s["wer_pct"],
            "w4a8_vs_w8a8": {
                **paired_bootstrap_delta(v_w8, v_w4, cluster_grouped=False),
                **pair_outcomes(v_w8, v_w4),
            },
            "w4a8_vs_w8a8_speaker_clustered": paired_bootstrap_delta(
                v_w8_spk, v_w4_spk, cluster_grouped=True
            ),
            "w4a8_vs_fp32": {
                **paired_bootstrap_delta(v_fp, v_w4, cluster_grouped=False),
                **pair_outcomes(v_fp, v_w4),
            },
            "w4a8_vs_fp32_speaker_clustered": paired_bootstrap_delta(
                v_fp_spk, v_w4_spk, cluster_grouped=True
            ),
            "w8a8_vs_fp32": {
                **paired_bootstrap_delta(v_fp, v_w8, cluster_grouped=False),
                **pair_outcomes(v_fp, v_w8),
            },
            "w8a8_vs_fp32_speaker_clustered": paired_bootstrap_delta(
                v_fp_spk, v_w8_spk, cluster_grouped=True
            ),
        },
        "vivos_clean_all_models_utt_vs_speaker": all_models_report,
    }
    out_path = BUILD_DIR / "wer_w4_vs_w8_paired_bootstrap.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
