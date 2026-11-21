#!/usr/bin/env python3
"""Retrospective 36-FFN (72-layer) RTL schedule verification against closed-loop FPGA logs.

Uses the untouched locked RTL predictor `predict_layer_rtl` (zero fitted parameters,
nominal 80/100 MHz clocks, REF_INTERVAL=1400, T_RFC=8), actual 128-bit SDRAM base
addresses from build/ffn_30m_sdram_image.json, and exact per-stage tile counts
(including 1..3 frame tail padding) from build/closed_loop_host_fpga_results.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_oos_stall_validation import predict_layer_rtl

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"


def main() -> int:
    img = json.loads((BUILD_DIR / "ffn_30m_sdram_image.json").read_text(encoding="utf-8"))
    cl = json.loads((BUILD_DIR / "closed_loop_host_fpga_results.json").read_text(encoding="utf-8"))
    g5b_36 = json.loads((BUILD_DIR / "g5b_30m_sdram_results.json").read_text(encoding="utf-8"))

    # Nominal 80 MHz engine / 100 MHz SDRAM schedule constants matching run_oos_stall_validation.py
    nom_eng_mhz = 80.0
    nom_sdram_mhz = 100.0
    ref_interval = 1400
    t_rfc = 8

    # Precompute per-block predicted cycles per T_tile=4 launch using exact SDRAM layout
    per_block_pred = []
    for blk in img["blocks"]:
        st_idx = int(blk["prefix"].split("/")[2])
        p1 = predict_layer_rtl(
            blk["l1_sdram_base_w128"],
            blk["l1_meta"],
            nom_sdram_mhz=nom_sdram_mhz,
            ref_interval=ref_interval,
            t_rfc=t_rfc,
            eng_mhz=nom_eng_mhz,
        )
        p2 = predict_layer_rtl(
            blk["l2_sdram_base_w128"],
            blk["l2_meta"],
            nom_sdram_mhz=nom_sdram_mhz,
            ref_interval=ref_interval,
            t_rfc=t_rfc,
            eng_mhz=nom_eng_mhz,
        )
        per_block_pred.append({
            "block_idx": blk["block_idx"],
            "stage": f"stage_{st_idx}",
            "l1": p1,
            "l2": p2,
            "block_total_cyc_per_tile": p1["pred_total_cyc"] + p2["pred_total_cyc"],
            "block_stall_cyc_per_tile": p1["pred_stall_cyc"] + p2["pred_stall_cyc"],
            "block_active_cyc_per_tile": p1["pred_active_cyc"] + p2["pred_active_cyc"],
        })

    results = []
    for utt in cl["utterances"]:
        pred_total_cyc = 0
        pred_stall_cyc = 0
        pred_active_cyc = 0
        for b in per_block_pred:
            n_t = int(utt["stage_shapes"][b["stage"]]["n_t_tiles"])
            pred_total_cyc += n_t * b["block_total_cyc_per_tile"]
            pred_stall_cyc += n_t * b["block_stall_cyc_per_tile"]
            pred_active_cyc += n_t * b["block_active_cyc_per_tile"]

        meas_total_cyc = int(utt["hw_total_cycles"])
        abs_err_cyc = pred_total_cyc - meas_total_cyc
        rel_err_pct = 100.0 * abs_err_cyc / float(meas_total_cyc)
        results.append({
            "utt_id": utt["utt_id"],
            "fbank_frames": utt["fbank_frames"],
            "duration_s": utt["duration_s"],
            "total_block_tiles": utt["total_block_tiles"],
            "meas_hw_total_cycles": meas_total_cyc,
            "pred_total_cycles_72layers": pred_total_cyc,
            "abs_cycle_error": abs_err_cyc,
            "rel_error_pct": round(rel_err_pct, 5),
            "pred_active_cycles": pred_active_cyc,
            "pred_stall_cycles": pred_stall_cyc,
        })

    # Per-block 1-tile (T_tile=4) retrospective verification against G5b 36-block hardware log
    per_block_1tile = []
    for bm, bp in zip(g5b_36["blocks"], per_block_pred):
        assert bm["block_idx"] == bp["block_idx"]
        blk_meta = img["blocks"][bm["block_idx"]]
        assert bm["prefix"] == blk_meta["prefix"]
        assert blk_meta["l1_sdram_base_w128"] % 32 == 0 and blk_meta["l2_sdram_base_w128"] % 32 == 0
        meas_c = int(bm["total_cycles"])
        pred_c = int(bp["block_total_cyc_per_tile"])
        rel_abs = 100.0 * abs(pred_c - meas_c) / float(meas_c)
        per_block_1tile.append({
            "block_idx": int(bm["block_idx"]),
            "prefix": bm["prefix"],
            "shape": bm["shape"],
            "meas_total_cyc": meas_c,
            "pred_total_cyc": pred_c,
            "abs_err_cyc": pred_c - meas_c,
            "rel_abs_err_pct": round(rel_abs, 5),
        })
    abs_errs = [r["rel_abs_err_pct"] for r in per_block_1tile]
    worst_blk = max(per_block_1tile, key=lambda r: r["rel_abs_err_pct"])
    per_block_summary = {
        "source_log": "build/g5b_30m_sdram_results.json",
        "num_blocks": len(per_block_1tile),
        "distinct_ffn_shapes": sorted({r["shape"] for r in per_block_1tile}),
        "all_layer_starts_mod32_zero": True,
        "median_abs_rel_err_pct": round(statistics.median(abs_errs), 5),
        "mean_abs_rel_err_pct": round(sum(abs_errs) / len(abs_errs), 5),
        "max_abs_rel_err_pct": round(max(abs_errs), 5),
        "worst_block": worst_blk,
        "blocks": per_block_1tile,
    }

    out = {
        "description": (
            "Retrospective full-encoder 36-FFN (72-layer) RTL schedule verification against "
            "closed-loop utterance counters and G5b per-block 1-tile counters at nominal 80/100 MHz (zero fitted parameters)."
        ),
        "nominal_clocks_mhz": {"eng": nom_eng_mhz, "sdram": nom_sdram_mhz},
        "sdram_refresh_cfg": {"ref_interval": ref_interval, "t_rfc": t_rfc},
        "utterances": results,
        "per_block_1tile_g5b": per_block_summary,
    }

    out_path = BUILD_DIR / "retrospective_36ffn_cycle_prediction.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    for r in results:
        print(
            f"  {r['utt_id']} (T={r['fbank_frames']}): meas={r['meas_hw_total_cycles']:,} "
            f"vs pred={r['pred_total_cycles_72layers']:,} (err={r['abs_cycle_error']:+,} cyc, {r['rel_error_pct']:+.5f}%)"
        )
    print(
        f"  36-block 1-tile G5b: median={per_block_summary['median_abs_rel_err_pct']:.5f}%, "
        f"mean={per_block_summary['mean_abs_rel_err_pct']:.5f}%, max={per_block_summary['max_abs_rel_err_pct']:.5f}% "
        f"(worst blk #{worst_blk['block_idx']} {worst_blk['shape']}: pred={worst_blk['pred_total_cyc']} vs meas={worst_blk['meas_total_cyc']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
