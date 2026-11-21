#!/usr/bin/env python3
"""B1.1: Expanded Closed-Loop Host<->FPGA Zipformer-30M Validation across
24 VIVOS Test utterances spanning all 19 speakers, diverse Fbank lengths
(141..491 frames, 1.406s..4.906s), and all stage_0 tail-padding mod-4 cases (0,1,2,3).

Saves incrementally after every utterance to build/closed_loop_24utts_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpga_jtag_runner import UsbBlasterJtag
from run_closed_loop_host_fpga import (
    BITEXACT_W4_ONNX,
    BUILD_DIR,
    MODEL_DIR,
    VIVOS_TEST_DIR,
    build_37_host_segments,
    load_tokens,
    load_vivos_prompts,
    measure_pll_clocks,
    pack_all_36_ffn_image,
    run_single_utterance_closed_loop,
    verify_or_upload_sdram,
)

UTTS_24 = [
    ("VIVOSDEV03_R119", "VIVOSDEV03"),
    ("VIVOSDEV03_R149", "VIVOSDEV03"),
    ("VIVOSDEV03_R163", "VIVOSDEV03"),
    ("VIVOSDEV08_274",  "VIVOSDEV08"),
    ("VIVOSDEV11_122",  "VIVOSDEV11"),
    ("VIVOSDEV09_091",  "VIVOSDEV09"),
    ("VIVOSDEV10_234",  "VIVOSDEV10"),
    ("VIVOSDEV11_057",  "VIVOSDEV11"),
    ("VIVOSDEV07_236",  "VIVOSDEV07"),
    ("VIVOSDEV02_R069", "VIVOSDEV02"),
    ("VIVOSDEV05_109",  "VIVOSDEV05"),
    ("VIVOSDEV13_241",  "VIVOSDEV13"),
    ("VIVOSDEV04_R162", "VIVOSDEV04"),
    ("VIVOSDEV12_166",  "VIVOSDEV12"),
    ("VIVOSDEV06_037",  "VIVOSDEV06"),
    ("VIVOSDEV14_186",  "VIVOSDEV14"),
    ("VIVOSDEV18_017",  "VIVOSDEV18"),
    ("VIVOSDEV15_020",  "VIVOSDEV15"),
    ("VIVOSDEV01_R098", "VIVOSDEV01"),
    ("VIVOSDEV19_029",  "VIVOSDEV19"),
    ("VIVOSDEV13_239",  "VIVOSDEV13"),
    ("VIVOSDEV17_173",  "VIVOSDEV17"),
    ("VIVOSDEV16_101",  "VIVOSDEV16"),
    ("VIVOSDEV16_182",  "VIVOSDEV16"),
]

OUT_JSON = BUILD_DIR / "closed_loop_24utts_results.json"


def summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    exact_enc = sum(1 for r in results if r["encoder_out_max_abs_diff"] == 0.0)
    exact_txt = sum(1 for r in results if r["transcript_exact_match_vs_onnx"] is True)
    tot_tiles = sum(r["total_block_tiles"] for r in results)
    tot_tail_launches = sum(r["padded_tail_block_launches"] for r in results)
    tot_byte_mism = sum(r["y_out_byte_mismatches"] or 0 for r in results)
    tot_csum_mism = sum(r["csum_mismatches"] or 0 for r in results)
    max_lsb = max(r["y_out_max_abs_diff_lsb"] or 0 for r in results)
    max_enc_diff = max(r["encoder_out_max_abs_diff"] for r in results)
    tot_audio_s = sum(r["duration_s"] for r in results)
    tot_fpga_s = sum(r["timing_breakdown_s"]["t_fpga_compute_s"] for r in results)
    tot_wall_s = sum(r["timing_breakdown_s"]["t_wall_audio_to_transcript_s"] for r in results)
    mod4_dist = {}
    for r in results:
        m = str(r["stage_shapes"]["stage_0"]["mod4"])
        mod4_dist[m] = mod4_dist.get(m, 0) + 1
    return {
        "n_utterances": n,
        "n_speakers": len({u.split("_")[0] for u in [r["utt_id"] for r in results]}),
        "total_audio_s": round(tot_audio_s, 3),
        "total_block_tiles": tot_tiles,
        "total_padded_tail_block_launches": tot_tail_launches,
        "stage0_mod4_counts": mod4_dist,
        "exact_encoder_out_matches": f"{exact_enc}/{n}",
        "exact_transcript_matches": f"{exact_txt}/{n}",
        "total_csum_mismatches": tot_csum_mism,
        "total_y_out_byte_mismatches": tot_byte_mism,
        "max_y_out_abs_diff_lsb": max_lsb,
        "max_encoder_out_abs_diff": max_enc_diff,
        "total_fpga_compute_s": round(tot_fpga_s, 4),
        "total_wall_s": round(tot_wall_s, 2),
        "aggregate_rtf_ffn": round(tot_fpga_s / tot_audio_s, 5),
        "aggregate_rtf_wall": round(tot_wall_s / tot_audio_s, 2),
        "mean_utt_wer_pct": round(float(np.mean([r["utt_wer_pct"] for r in results])), 2),
    }


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_vivos_prompts()
    tokens = load_tokens()
    all_words, manifest, _ = pack_all_36_ffn_image()
    segments, mono_sess = build_37_host_segments(BITEXACT_W4_ONNX, manifest)

    dec_opts = ort.SessionOptions(); dec_opts.intra_op_num_threads = 1
    jn_opts = ort.SessionOptions(); jn_opts.intra_op_num_threads = 1
    dec_sess = ort.InferenceSession(str(MODEL_DIR / "decoder-epoch-20-avg-10.onnx"), sess_options=dec_opts, providers=["CPUExecutionProvider"])
    jn_sess = ort.InferenceSession(str(MODEL_DIR / "joiner-epoch-20-avg-10.onnx"), sess_options=jn_opts, providers=["CPUExecutionProvider"])

    results: list[dict] = []
    with UsbBlasterJtag() as dev:
        cap = dev.read_resp(0x2005)
        assert (cap >> 32) & 1 == 1, "SDRAM not ready!"
        eng_mhz, sd_mhz, _, _ = measure_pll_clocks(dev)
        already_ok, up_s = verify_or_upload_sdram(dev, all_words, manifest)
        print(f"[24-Utt Closed-Loop] eng={eng_mhz} MHz, sdram={sd_mhz} MHz | resident_reused={already_ok} (upload={up_s:.1f}s)")

        for idx, (uid, spk) in enumerate(UTTS_24, 1):
            wp = VIVOS_TEST_DIR / "waves" / spk / f"{uid}.wav"
            res = run_single_utterance_closed_loop(
                wp, segments, manifest, dev, dec_sess, jn_sess, tokens, mono_sess,
                prompts[uid], verify_golden=True,
            )
            exact_fpga_s = res["hw_total_cycles"] / (eng_mhz * 1e6)
            tb = res["timing_breakdown_s"]
            diff_eng = exact_fpga_s - tb["t_fpga_compute_s"]
            tb["t_fpga_compute_s"] = round(exact_fpga_s, 4)
            tb["t_ctrl_lut_sync_s"] = round(max(0.0, tb["t_ctrl_lut_sync_s"] - diff_eng), 3)
            res["speaker"] = spk
            res["measured_rtf_ffn"] = round(exact_fpga_s / res["duration_s"], 5)
            res["measured_rtf_wall"] = round(tb["t_wall_audio_to_transcript_s"] / res["duration_s"], 2)
            res["mac_lane_util_pct"] = round(100.0 * res["hw_mac_cycles"] / res["hw_total_cycles"], 2)
            res["stall_pct"] = round(100.0 * res["hw_stall_cycles"] / res["hw_total_cycles"], 2)
            results.append(res)

            out = {
                "experiment": "B1_1_Closed_Loop_24_VIVOS_Utterances",
                "device": "EP4CE10E22C8N",
                "eng_mhz": eng_mhz,
                "sdram_mhz": sd_mhz,
                "sdram_image_reused": already_ok,
                "summary": summarize(results),
                "utterances": results,
            }
            OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(
                f"[{idx:02d}/24] {uid:16s} ({res['duration_s']:5.3f}s, T={res['fbank_frames']:3d}, "
                f"s0_mod4={res['stage_shapes']['stage_0']['mod4']}, tiles={res['total_block_tiles']:3d}): "
                f"fpga={exact_fpga_s*1e3:6.1f}ms, wall={tb['t_wall_audio_to_transcript_s']:6.2f}s, "
                f"enc_diff={res['encoder_out_max_abs_diff']}, match={res['transcript_exact_match_vs_onnx']}"
            )

    print(f"\nCompleted 24/24 utterances -> {OUT_JSON}")
    print(json.dumps(summarize(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
