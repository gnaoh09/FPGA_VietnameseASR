#!/usr/bin/env python3
"""Run RTL simulation & post-hoc model analyses B1..B4:
  B1: Summarize 66-run offset sim at real measured clocks 79.956 / 89.994 MHz (from analysis_41_42_results.json).
  B2: Simulate 60/60 MHz across ppm offsets {0, +100, -100, +200, -200, +400 (measured), -400, +800, -800} ppm.
  B3: Post-hoc 1:1 CDC extension (+c_1to1 per packet when |f_eng - f_sdram|/f_sdram < 0.01) on 60-pt OOS.
  B4: Fine cold-start refresh-phase sweep (0..1250 step 25 = 51 phases) on L1 (offset 1 vs 15) @ 79.956/89.994 MHz,
      tracking exact ref_cnt & first-fill vs subsequent refresh collisions.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_deep_rtl_validation_part2 import SHAPES_6, predict_layer_param
from run_oos_stall_validation import make_synthetic_integer_ffn

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OUT_JSON = BUILD_DIR / "b1_b4_rtl_sim_results.json"


def run_b2_ppm_sweep() -> list[dict]:
    d2 = json.loads((BUILD_DIR / "deep_rtl_validation_part2.json").read_text())
    e60_recs = [r for r in d2["records"] if r["tag"] == "e60_s60"]
    wl_map = {s["name"]: make_synthetic_integer_ffn(s["d_model"], s["d_ff"], seed=20261130 + i) for i, s in enumerate(SHAPES_6)}

    vvp60 = BUILD_DIR / "tb_78pt_sd60.vvp"
    if not vvp60.exists():
        subprocess.run(
            ["iverilog", "-g2012", "-DSIMULATION", "-Ptb_78pt_sim.CLK_MHZ_P=60",
             "-o", str(vvp60), "rtl/sdram_ctrl.v", "rtl/zipformer_engine.v", "tb/tb_78pt_sim.v"],
            cwd=str(ROOT_DIR), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    rows = []
    # Test measured 59.969 / 59.993 (~+400 ppm sdram faster) plus synthetic ppm offsets around 60.000 MHz
    ppm_cases = [
        ("0ppm (60.000/60.000)", 60.000, 60.000),
        ("-400ppm (60.000/59.976)", 60.000, 59.976),
        ("-200ppm (60.000/59.988)", 60.000, 59.988),
        ("+200ppm (60.000/60.012)", 60.000, 60.012),
        ("+400ppm_meas (59.969/59.993)", 59.969, 59.993),
        ("+800ppm (60.000/60.048)", 60.000, 60.048),
        ("-800ppm (60.000/59.952)", 60.000, 59.952),
    ]
    for label, f_eng, f_sd in ppm_cases:
        eh = 500.0 / f_eng
        sh = 500.0 / f_sd
        errs_vs_sil = []
        diffs_vs_0ppm = []
        for r in e60_recs:
            w = wl_map[r["shape"]]
            m1, m2 = w["l1_meta"], w["l2_meta"]
            cmd = [
                "vvp", str(vvp60),
                f"+ENG_HALF_NS={eh:.6f}", f"+SDRAM_HALF_NS={sh:.6f}",
                f"+L1_KC={m1['k_chunk']}", f"+L1_NT={m1['n_tiles']}", f"+L1_KCH={m1['k_chunks']}",
                f"+L1_WW={m1['w_words_per_chunk']}", f"+L1_BASE={r['l1_base']}",
                f"+L2_KC={m2['k_chunk']}", f"+L2_NT={m2['n_tiles']}", f"+L2_KCH={m2['k_chunks']}",
                f"+L2_WW={m2['w_words_per_chunk']}", f"+L2_BASE={r['l2_base']}",
                f"+DM={w['d_model']}", f"+DF={w['d_ff']}",
            ]
            out = subprocess.check_output(cmd, cwd=str(ROOT_DIR), text=True)
            m = re.search(r"\btot=(\d+)", out)
            assert m
            sim_t = int(m.group(1))
            errs_vs_sil.append(abs(sim_t - r["meas_tot_mean"]) / r["meas_tot_mean"] * 100.0)
            diffs_vs_0ppm.append(sim_t)
        rows.append({
            "case": label,
            "eng_mhz": f_eng,
            "sdram_mhz": f_sd,
            "mean_err_vs_silicon_pct": round(float(np.mean(errs_vs_sil)), 4),
            "sim_totals_6shapes": diffs_vs_0ppm,
        })
    return rows


def run_b3_posthoc_1to1_cdc() -> dict:
    d2 = json.loads((BUILD_DIR / "deep_rtl_validation_part2.json").read_text())
    oos = [r for r in d2["records"] if r["group"] == "oos_60pt"]
    wl_map = {s["name"]: make_synthetic_integer_ffn(s["d_model"], s["d_ff"], seed=20261130 + i) for i, s in enumerate(SHAPES_6)}

    # Post-hoc 1:1 CDC term: when |f_eng - f_sdram| / f_sdram < 0.01 (i.e. 80/80 MHz),
    # add delta_cdc_1to1 per DMA packet (test c_1to1 = 1.0 cyc/pkt from RTL toggle-CDC phase lock)
    c_1to1 = 1.0
    orig_errs, post_errs, orig_bias, post_bias = [], [], [], []
    e80_orig, e80_post = [], []
    for r in oos:
        w = wl_map[r["shape"]]
        M_tot = w["l1_meta"]["n_tiles"] * w["l1_meta"]["k_chunks"] + w["l2_meta"]["n_tiles"] * w["l2_meta"]["k_chunks"]
        is_1to1 = abs(r["eng_mhz"] - r["sdram_mhz"]) / float(r["sdram_mhz"]) < 0.01
        pred_orig = r["wb_pred"]
        pred_post = pred_orig + (c_1to1 * M_tot if is_1to1 else 0.0)
        e_o = abs(pred_orig - r["meas_tot_mean"]) / r["meas_tot_mean"] * 100.0
        e_p = abs(pred_post - r["meas_tot_mean"]) / r["meas_tot_mean"] * 100.0
        orig_errs.append(e_o); post_errs.append(e_p)
        orig_bias.append(pred_orig - r["meas_tot_mean"])
        post_bias.append(pred_post - r["meas_tot_mean"])
        if is_1to1:
            e80_orig.append({"shape": r["shape"], "err_pct": round(e_o, 5), "cyc_err": round(pred_orig - r["meas_tot_mean"], 2)})
            e80_post.append({"shape": r["shape"], "err_pct": round(e_p, 5), "cyc_err": round(pred_post - r["meas_tot_mean"], 2)})

    return {
        "note": "Post-hoc extension (+1.0 cyc/packet when |f_eng - f_sdram|/f_sdram < 1%); separate from frozen held-out",
        "e80_s80_orig": e80_orig,
        "e80_s80_posthoc": e80_post,
        "oos_60pt_orig": {
            "mean_pct": round(float(np.mean(orig_errs)), 5),
            "median_pct": round(float(np.median(orig_errs)), 5),
            "p95_pct": round(float(np.percentile(orig_errs, 95)), 5),
            "max_pct": round(float(np.max(orig_errs)), 5),
            "mean_bias_cyc": round(float(np.mean(orig_bias)), 2),
        },
        "oos_60pt_posthoc": {
            "mean_pct": round(float(np.mean(post_errs)), 5),
            "median_pct": round(float(np.median(post_errs)), 5),
            "p95_pct": round(float(np.percentile(post_errs, 95)), 5),
            "max_pct": round(float(np.max(post_errs)), 5),
            "mean_bias_cyc": round(float(np.mean(post_bias)), 2),
        },
    }


def run_b4_cold_start_l1() -> dict:
    # Fine 51-phase sweep (0..1250 step 25) on L1 only (offset 1 vs 15) @ 79.956/89.994 MHz
    vvp_90 = BUILD_DIR / "tb_41_offset90.vvp"
    eng_half = 500.0 / 79.956
    sd_half = 500.0 / 89.994
    dl1_list, l1_1_list, l1_15_list = [], [], []
    for ref_d in range(0, 1251, 25):
        out = subprocess.check_output(
            ["vvp", str(vvp_90), f"+ENG_HALF={eng_half:.6f}", f"+SD_HALF={sd_half:.6f}",
             "+INIT_SD_DELAY=0.0", f"+REF_DELAY={ref_d}"],
            cwd=str(ROOT_DIR), text=True
        )
        m = re.search(r"RES l1_1=(\d+) l2_1=(\d+) l1_15=(\d+) l2_15=(\d+) dl1=(-?\d+)", out)
        assert m
        l1_1_list.append(int(m.group(1)))
        l1_15_list.append(int(m.group(3)))
        dl1_list.append(int(m.group(5)))
    return {
        "n_phases": len(dl1_list),
        "l1_off1_mean_sd": [round(float(np.mean(l1_1_list)), 2), round(float(np.std(l1_1_list, ddof=1)), 2)],
        "l1_off1_min_max": [min(l1_1_list), max(l1_1_list)],
        "l1_off15_mean_sd": [round(float(np.mean(l1_15_list)), 2), round(float(np.std(l1_15_list, ddof=1)), 2)],
        "l1_off15_min_max": [min(l1_15_list), max(l1_15_list)],
        "dl1_mean_sd": [round(float(np.mean(dl1_list)), 2), round(float(np.std(dl1_list, ddof=1)), 2)],
        "dl1_min_max": [min(dl1_list), max(dl1_list)],
        "pos_count": int(sum(1 for x in dl1_list if x > 0)),
    }


def main() -> int:
    a41 = json.loads((BUILD_DIR / "analysis_41_42_results.json").read_text())
    res = {
        "B1_offset_realclk_66runs": a41["sec_4_1"]["offset_realclk_66runs"],
        "B2_60_60_ppm_sweep": run_b2_ppm_sweep(),
        "B3_posthoc_1to1_cdc": run_b3_posthoc_1to1_cdc(),
        "B4_cold_start_l1_51phases": run_b4_cold_start_l1(),
    }
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
