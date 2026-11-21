#!/usr/bin/env python3
"""Evaluate the 12-point out-of-sample (3 unseen FFN shapes x 4 SDRAM frequencies)
cycle-prediction baseline (measured BIST bandwidth roofline) and component ablations
from existing silicon logs in build/oos_stall_validation_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_oos_stall_validation import (
    FREQ_PLAN,
    UNSEEN_SHAPES,
    make_synthetic_integer_ffn,
    predict_layer_rtl,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"


def predict_layer_ablation(
    base_w128: int,
    meta: dict,
    nom_sdram_mhz: float,
    ref_interval: int,
    t_rfc: int,
    eng_mhz: float = 80.0,
    include_pages: bool = True,
    include_cold_start: bool = True,
    include_refresh: bool = True,
    use_mean_pages: bool = False,
) -> int:
    """Parameterized variant of predict_layer_rtl for component ablation."""
    n_o = int(meta["n_tiles"])
    k_chunks = int(meta["k_chunks"])
    k_chunk = int(meta["k_chunk"])
    pkt_words = int(meta["pkt_words"])
    num_chunks = n_o * k_chunks

    pages_seq: list[float] = []
    addr = base_w128
    for _ in range(num_chunks):
        pages_seq.append(
            float(((addr + pkt_words - 1) // 32) - (addr // 32) + 1) if include_pages else 0.0
        )
        addr += pkt_words
    if use_mean_pages and include_pages:
        mean_p = sum(pages_seq) / float(num_chunks)
        pages_seq = [mean_p] * num_chunks

    c_tiles: list[int] = []
    c_dmas: list[float] = []
    ref_mul = (1.0 + t_rfc / float(ref_interval)) if include_refresh else 1.0
    p_idx = 0
    for _no in range(n_o):
        for kc in range(k_chunks):
            has_epi = (kc == k_chunks - 1)
            c_tile = 4 * k_chunk + (13 if has_epi else 6)
            sd_cyc = (8 * pkt_words + 14.0 * pages_seq[p_idx] + 1.0) * ref_mul + 3.0
            c_dma = sd_cyc * (eng_mhz / nom_sdram_mhz) + 4.0
            c_tiles.append(c_tile)
            c_dmas.append(c_dma)
            p_idx += 1

    stall = c_dmas[0] if include_cold_start else 0.0
    for idx in range(1, num_chunks):
        stall += max(0.0, c_dmas[idx] - c_tiles[idx - 1])
    return int(round(float(sum(c_tiles)) + stall))


def bist_roofline_layer_cycles(meta: dict, bist_rd_mbps: float, eng_mhz: float = 80.0) -> float:
    """Simple BIST roofline per layer: max(compute_cycles_with_epilogue, total_bytes / BIST_BW)."""
    n_o = int(meta["n_tiles"])
    k_chunks = int(meta["k_chunks"])
    k_chunk = int(meta["k_chunk"])
    c_comp = n_o * ((k_chunks - 1) * (4 * k_chunk + 6) + (4 * k_chunk + 13))
    c_bist = (int(meta["total_bytes"]) / (bist_rd_mbps * 1e6)) * (eng_mhz * 1e6)
    return max(float(c_comp), c_bist)


def main() -> int:
    oos = json.loads((BUILD_DIR / "oos_stall_validation_results.json").read_text(encoding="utf-8"))

    workloads = []
    offset = 0
    for idx, spec in enumerate(UNSEEN_SHAPES):
        wl = make_synthetic_integer_ffn(spec["d_model"], spec["d_ff"], seed=20261119 + idx)
        l1_base = offset
        offset += len(wl["l1_words"])
        l2_base = offset
        offset += len(wl["l2_words"])
        workloads.append({
            **spec,
            "l1_meta": wl["l1_meta"],
            "l2_meta": wl["l2_meta"],
            "l1_base": l1_base,
            "l2_base": l2_base,
        })

    model_specs = [
        ("bist_bandwidth_roofline", "Compute + BIST measured read bandwidth roofline", None),
        ("no_pages_no_coldstart", "RTL model without page-open (14*p_i=0) and without cold-start DMA", (False, False, True, False)),
        ("no_coldstart_only", "RTL model without cold-start first DMA (keep 14*p_i)", (True, False, True, False)),
        ("no_pages_only", "RTL model without page-open penalty (14*p_i=0, keep cold-start)", (False, True, True, False)),
        ("no_refresh_only", "RTL model without average refresh multiplier (T_RFC=0)", (True, True, False, False)),
        ("mean_pages_within_layer", "RTL model using within-layer mean pages per packet", (True, True, True, True)),
        ("full_locked_rtl_model", "Full locked RTL schedule model (paper)", (True, True, True, False)),
    ]

    summary: dict[str, dict] = {k: {"description": d, "points": []} for k, d, _ in model_specs}
    per_point = []

    for fp in FREQ_PLAN:
        f_mhz = int(fp["nom_sdram_mhz"])
        fk = f"sdram_{f_mhz}mhz"
        f_entry = oos["per_frequency"][fk]
        bist_mbps = float(f_entry["bist_1mib_rd_MBps"])

        for wl in workloads:
            sname = wl["name"]
            meas_tot = int(f_entry["shapes"][sname]["combined"]["meas_total_cyc"])
            bist_tot_flt = (
                bist_roofline_layer_cycles(wl["l1_meta"], bist_mbps)
                + bist_roofline_layer_cycles(wl["l2_meta"], bist_mbps)
            )
            bist_tot = int(round(bist_tot_flt))

            pt_row = {
                "nom_sdram_mhz": f_mhz,
                "bist_1mib_rd_MBps": bist_mbps,
                "shape": sname,
                "meas_total_cyc": meas_tot,
                "predictions": {"bist_bandwidth_roofline": bist_tot},
            }

            for m_key, _desc, flags in model_specs:
                if flags is None:
                    pred_exact = bist_tot_flt
                    pred = bist_tot
                else:
                    inc_p, inc_c, inc_r, use_mp = flags
                    pred = (
                        predict_layer_ablation(
                            wl["l1_base"], wl["l1_meta"], f_mhz, fp["ref_interval"], fp["t_rfc"],
                            include_pages=inc_p, include_cold_start=inc_c, include_refresh=inc_r, use_mean_pages=use_mp,
                        )
                        + predict_layer_ablation(
                            wl["l2_base"], wl["l2_meta"], f_mhz, fp["ref_interval"], fp["t_rfc"],
                            include_pages=inc_p, include_cold_start=inc_c, include_refresh=inc_r, use_mean_pages=use_mp,
                        )
                    )
                    pred_exact = float(pred)
                pt_row["predictions"][m_key] = pred
                rel_signed_pct = 100.0 * (pred_exact - meas_tot) / float(meas_tot)
                rel_abs_pct = abs(rel_signed_pct)
                summary[m_key]["points"].append({
                    "nom_sdram_mhz": f_mhz,
                    "shape": sname,
                    "meas_total_cyc": meas_tot,
                    "pred_total_cyc": pred,
                    "rel_signed_err_pct": round(rel_signed_pct, 5),
                    "rel_abs_err_pct": round(rel_abs_pct, 5),
                    "_raw_rel_abs_pct": rel_abs_pct,
                })

            per_point.append(pt_row)

    summary_out = {}
    for m_key, blk in summary.items():
        pts = blk["points"]
        raw_errs = [p.pop("_raw_rel_abs_pct") for p in pts]
        worst = max(pts, key=lambda p: p["rel_abs_err_pct"])
        summary_out[m_key] = {
            "description": blk["description"],
            "mean_abs_rel_err_pct": round(sum(raw_errs) / len(raw_errs), 5),
            "mean_abs_rel_err_pct_4dp": round(sum(raw_errs) / len(raw_errs), 4),
            "max_abs_rel_err_pct": round(max(raw_errs), 5),
            "max_abs_rel_err_pct_4dp": round(max(raw_errs), 4),
            "worst_case": worst,
            "example_90mhz_192_384_192": next(
                p for p in pts if p["nom_sdram_mhz"] == 90 and p["shape"] == "ffn_192_384_192"
            ),
        }

    # Retrospective evaluation on the 12 measured points:
    # lowest SDRAM freq in {60, 90, 100, 120} MHz within 1.0% of 120 MHz latency
    freq_selection = {}
    for wl in workloads:
        sname = wl["name"]
        s_pts = {p["nom_sdram_mhz"]: p for p in per_point if p["shape"] == sname}
        m120 = float(s_pts[120]["meas_total_cyc"])
        r120 = float(s_pts[120]["predictions"]["full_locked_rtl_model"])
        mp120 = float(s_pts[120]["predictions"]["mean_pages_within_layer"])
        b120 = float(s_pts[120]["predictions"]["bist_bandwidth_roofline"])
        per_f = {}
        for f in [60, 90, 100, 120]:
            per_f[str(f)] = {
                "meas_vs_120_pct": round(100.0 * (s_pts[f]["meas_total_cyc"] - m120) / m120, 2),
                "rtl_vs_120_pct": round(100.0 * (s_pts[f]["predictions"]["full_locked_rtl_model"] - r120) / r120, 2),
                "mean_pages_vs_120_pct": round(100.0 * (s_pts[f]["predictions"]["mean_pages_within_layer"] - mp120) / mp120, 2),
                "bist_vs_120_pct": round(100.0 * (s_pts[f]["predictions"]["bist_bandwidth_roofline"] - b120) / b120, 2),
            }
        sel_meas = min(f for f in [60, 90, 100, 120] if per_f[str(f)]["meas_vs_120_pct"] <= 1.0)
        sel_rtl = min(f for f in [60, 90, 100, 120] if per_f[str(f)]["rtl_vs_120_pct"] <= 1.0)
        sel_mp = min(f for f in [60, 90, 100, 120] if per_f[str(f)]["mean_pages_vs_120_pct"] <= 1.0)
        sel_bist = min(f for f in [60, 90, 100, 120] if per_f[str(f)]["bist_vs_120_pct"] <= 1.0)
        freq_selection[sname] = {
            "evaluation_note": "Retrospective check of 1% decision rule on 12 measured points",
            "silicon_selected_mhz": sel_meas,
            "full_rtl_selected_mhz": sel_rtl,
            "mean_pages_selected_mhz": sel_mp,
            "bist_roofline_selected_mhz": sel_bist,
            "full_rtl_matches_silicon": sel_rtl == sel_meas,
            "mean_pages_matches_silicon": sel_mp == sel_meas,
            "bist_matches_silicon": sel_bist == sel_meas,
            "degradation_vs_120mhz": per_f,
        }

    # Pre-registered offset-shift predictions on 256->768->256 @ engine 80 / SDRAM 90 MHz
    wl_768 = make_synthetic_integer_ffn(256, 768, seed=20261120)
    l1_words_768 = len(wl_768["l1_words"])
    offset_rows_768 = []
    for off in [0, 1, 15, 31]:
        p1 = predict_layer_rtl(off, wl_768["l1_meta"], 90.0, 1260, 8, 80.0)
        p2 = predict_layer_rtl(off + l1_words_768, wl_768["l2_meta"], 90.0, 1260, 8, 80.0)
        mp1 = predict_layer_ablation(off, wl_768["l1_meta"], 90.0, 1260, 8, 80.0, use_mean_pages=True)
        mp2 = predict_layer_ablation(off + l1_words_768, wl_768["l2_meta"], 90.0, 1260, 8, 80.0, use_mean_pages=True)
        offset_rows_768.append({
            "offset_w128": off,
            "l1_base_w128": off,
            "l2_base_w128": off + l1_words_768,
            "l1_pred_cyc": p1["pred_total_cyc"],
            "l2_pred_cyc": p2["pred_total_cyc"],
            "combined_pred_cyc": p1["pred_total_cyc"] + p2["pred_total_cyc"],
            "mean_pages_pred_cyc": mp1 + mp2,
        })
    prereg_offset_768 = {
        "shape": "ffn_256_768_256",
        "clocks_mhz": {"eng": 80.0, "sdram": 90.0},
        "refresh_cfg": {"ref_interval": 1260, "t_rfc": 8},
        "t_tile": 4,
        "status": "pre_registered_prediction_not_yet_measured_on_board",
        "offsets": offset_rows_768,
        "delta_offset31_minus_offset0_cyc": offset_rows_768[-1]["combined_pred_cyc"] - offset_rows_768[0]["combined_pred_cyc"],
        "delta_offset31_minus_offset0_mean_pages_cyc": offset_rows_768[-1]["mean_pages_pred_cyc"] - offset_rows_768[0]["mean_pages_pred_cyc"],
    }

    # Discriminating offset-shift test: 192->640->192 @ engine 80 / SDRAM 90 MHz
    # Separates per-packet page counting (+42 cycles for offset 1 vs 15) from within-layer mean pages (0 cycles)
    wl_640 = make_synthetic_integer_ffn(192, 640, seed=20261121)
    l1_words_640 = len(wl_640["l1_words"])
    offset_rows_640 = []
    for off in [0, 1, 15, 31]:
        p1 = predict_layer_rtl(off, wl_640["l1_meta"], 90.0, 1260, 8, 80.0)
        p2 = predict_layer_rtl(off + l1_words_640, wl_640["l2_meta"], 90.0, 1260, 8, 80.0)
        mp1 = predict_layer_ablation(off, wl_640["l1_meta"], 90.0, 1260, 8, 80.0, use_mean_pages=True)
        mp2 = predict_layer_ablation(off + l1_words_640, wl_640["l2_meta"], 90.0, 1260, 8, 80.0, use_mean_pages=True)
        offset_rows_640.append({
            "offset_w128": off,
            "l1_base_w128": off,
            "l2_base_w128": off + l1_words_640,
            "l1_pred_cyc": p1["pred_total_cyc"],
            "l2_pred_cyc": p2["pred_total_cyc"],
            "combined_pred_cyc": p1["pred_total_cyc"] + p2["pred_total_cyc"],
            "l1_mean_pages_cyc": mp1,
            "l2_mean_pages_cyc": mp2,
            "mean_pages_pred_cyc": mp1 + mp2,
        })
    r640_by_off = {r["offset_w128"]: r for r in offset_rows_640}
    prereg_offset_640 = {
        "shape": "ffn_192_640_192",
        "clocks_mhz": {"eng": 80.0, "sdram": 90.0},
        "refresh_cfg": {"ref_interval": 1260, "t_rfc": 8},
        "t_tile": 4,
        "seed": 20261121,
        "l1_words_128b": l1_words_640,
        "l2_words_128b": len(wl_640["l2_words"]),
        "status": "locked_before_board_measurement",
        "offsets": offset_rows_640,
        "delta_offset1_minus_offset15_per_pkt_cyc": r640_by_off[1]["combined_pred_cyc"] - r640_by_off[15]["combined_pred_cyc"],
        "delta_offset1_minus_offset15_mean_pages_cyc": r640_by_off[1]["mean_pages_pred_cyc"] - r640_by_off[15]["mean_pages_pred_cyc"],
    }

    out = {
        "source_log": "build/oos_stall_validation_results.json",
        "num_points": len(per_point),
        "summary_12pt": summary_out,
        "frequency_selection_decision_1pct": freq_selection,
        "preregistered_offset_sweep_256_768_256_90mhz": prereg_offset_768,
        "preregistered_offset_sweep_192_640_192_90mhz": prereg_offset_640,
        "per_point": per_point,
    }
    out_path = BUILD_DIR / "oos_12pt_ablation_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("=== 12-POINT OOS BASELINE & ABLATION ===")
    for m_key, s in summary_out.items():
        ex = s["example_90mhz_192_384_192"]
        print(
            f"{m_key:26s} | mean={s['mean_abs_rel_err_pct']:.5f}% | max={s['max_abs_rel_err_pct']:.5f}% "
            f"| 90MHz_192={ex['pred_total_cyc']} (meas={ex['meas_total_cyc']}, {ex['rel_signed_err_pct']:+.2f}%)"
        )
    print("Freq selection 1% rule (silicon, RTL, mean_pages, BIST):", {k: (v["silicon_selected_mhz"], v["full_rtl_selected_mhz"], v["mean_pages_selected_mhz"], v["bist_roofline_selected_mhz"]) for k, v in freq_selection.items()})
    print("Preregistered 192->640->192 @90MHz (off 1 vs 15):", "per_pkt=", (r640_by_off[1]["combined_pred_cyc"], r640_by_off[15]["combined_pred_cyc"]), "diff=", prereg_offset_640["delta_offset1_minus_offset15_per_pkt_cyc"], "| mean_pages=", (r640_by_off[1]["mean_pages_pred_cyc"], r640_by_off[15]["mean_pages_pred_cyc"]), "diff=", prereg_offset_640["delta_offset1_minus_offset15_mean_pages_cyc"])
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
