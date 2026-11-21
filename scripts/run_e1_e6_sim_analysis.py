#!/usr/bin/env python3
"""Execute Round-4 RTL simulations & analyses E1..E6 (no board needed):
  E1: Refresh-disabled counterfactual of B4 (51 cold-start phases on L1 offset 1 vs 15 @ 80/90 MHz).
      Uses build/sdram_ctrl_noref.v (leaving rtl/sdram_ctrl.v bit-for-bit untouched).
  E2: 60/60 MHz 6-shape comparison: 0 ppm (exact same-VCO) vs +400 ppm (JTAG-noisy old sim) vs silicon.
  E3: Source of absolute 15-31 cycle sim-vs-silicon offset: exact 0 ppm 80.000/90.000 (66 runs) vs
      noisy 79.956/89.994 (66 runs) vs silicon, plus RTL counter start/stop audit.
  E4: Mechanistic analytical check of L1 (+8..+11 cyc): compare refresh-off vs refresh-on per-packet.
  E5: Residual regression AFTER B3 (+1.0 cyc/pkt at 1:1) by M, f_sdram, K_chunks on 60-pt set.
  E6: Apply B3 (+1.0 cyc/pkt) to 60/60 MHz (6 shapes) and all 1:1 conditions in the 78-pt set.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_deep_rtl_validation_part2 import SHAPES_6
from run_oos_stall_validation import make_synthetic_integer_ffn

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OUT_JSON = BUILD_DIR / "e1_e6_sim_analysis_results.json"


def run_e1_and_e3_offset_sims() -> dict:
    # Build refresh-disabled copy of sdram_ctrl.v strictly inside build/ for counterfactual E1
    src = (ROOT_DIR / "rtl" / "sdram_ctrl.v").read_text(encoding="utf-8")
    # Disable auto-refresh by forcing ref_pending <= 1'b0;
    patched = src.replace("ref_due   <= 1'b1;", "ref_due   <= 1'b0;")
    assert patched != src, "Failed to patch ref_due in sdram_ctrl.v"
    noref_path = BUILD_DIR / "sdram_ctrl_noref.v"
    noref_path.write_text(patched, encoding="utf-8")

    vvp_noref = BUILD_DIR / "tb_e1_noref90.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-DSIMULATION", "-o", str(vvp_noref),
         str(noref_path), "rtl/zipformer_engine.v", "tb/tb_41_offset90.v"],
        cwd=str(ROOT_DIR), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    vvp_ref = BUILD_DIR / "tb_41_offset90.vvp"

    # E1: Run 51 phases with refresh OFF at exact 80.000 / 90.000 MHz (0 ppm) AND at 79.956 / 89.994
    e1_res = {}
    for label, f_eng, f_sd in [("exact_80_90_0ppm", 80.0, 90.0), ("meas_79956_89994", 79.956, 89.994)]:
        eh, sh = 500.0 / f_eng, 500.0 / f_sd
        dl1s, dl2s, l1_1s, l1_15s, l2_1s, l2_15s = [], [], [], [], [], []
        for ref_d in range(0, 1251, 25):
            out = subprocess.check_output(
                ["vvp", str(vvp_noref), f"+ENG_HALF={eh:.6f}", f"+SD_HALF={sh:.6f}",
                 "+INIT_SD_DELAY=0.0", f"+REF_DELAY={ref_d}"],
                cwd=str(ROOT_DIR), text=True,
            )
            m = re.search(r"RES l1_1=(\d+) l2_1=(\d+) l1_15=(\d+) l2_15=(\d+) dl1=(-?\d+) dl2=(-?\d+)", out)
            assert m
            l1_1s.append(int(m.group(1))); l2_1s.append(int(m.group(2)))
            l1_15s.append(int(m.group(3))); l2_15s.append(int(m.group(4)))
            dl1s.append(int(m.group(5))); dl2s.append(int(m.group(6)))
        e1_res[label] = {
            "n_phases": len(dl1s),
            "l1_off1_mean": round(float(np.mean(l1_1s)), 2),
            "l1_off15_mean": round(float(np.mean(l1_15s)), 2),
            "dl1_mean": round(float(np.mean(dl1s)), 2),
            "dl1_sd": round(float(np.std(dl1s, ddof=1)), 2),
            "dl1_min_max": [min(dl1s), max(dl1s)],
            "dl2_mean": round(float(np.mean(dl2s)), 2),
            "dl2_min_max": [min(dl2s), max(dl2s)],
        }

    # E3: Exact 80.000 / 90.000 MHz (0 ppm same-VCO PLL) with refresh ON across 66 runs (11 ref x 6 init phase)
    eh0, sh0 = 500.0 / 80.0, 500.0 / 90.0
    dl1s, dl2s, dtots, o1s, o15s = [], [], [], [], []
    for ref_d in range(0, 1001, 100):
        for init_d in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
            out = subprocess.check_output(
                ["vvp", str(vvp_ref), f"+ENG_HALF={eh0:.6f}", f"+SD_HALF={sh0:.6f}",
                 f"+INIT_SD_DELAY={init_d:.3f}", f"+REF_DELAY={ref_d}"],
                cwd=str(ROOT_DIR), text=True,
            )
            m = re.search(r"RES l1_1=(\d+) l2_1=(\d+) l1_15=(\d+) l2_15=(\d+) dl1=(-?\d+) dl2=(-?\d+) dtot=(-?\d+)", out)
            assert m
            l1_1, l2_1, l1_15, l2_15 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            dl1s.append(int(m.group(5))); dl2s.append(int(m.group(6))); dtots.append(int(m.group(7)))
            o1s.append(l1_1 + l2_1); o15s.append(l1_15 + l2_15)

    # Also test 66 runs with refresh OFF at exact 80/90 across 11 ref x 6 init phase
    dl1_nr66 = []
    for ref_d in range(0, 1001, 100):
        for init_d in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
            out = subprocess.check_output(
                ["vvp", str(vvp_noref), f"+ENG_HALF={eh0:.6f}", f"+SD_HALF={sh0:.6f}",
                 f"+INIT_SD_DELAY={init_d:.3f}", f"+REF_DELAY={ref_d}"],
                cwd=str(ROOT_DIR), text=True,
            )
            m = re.search(r"dl1=(-?\d+)", out)
            assert m
            dl1_nr66.append(int(m.group(1)))

    e3_exact_80_90 = {
        "n_runs": len(dtots),
        "o1_mean": round(float(np.mean(o1s)), 2),
        "o15_mean": round(float(np.mean(o15s)), 2),
        "o1_gap_vs_silicon_65333_45": round(float(np.mean(o1s)) - 65333.45, 2),
        "o15_gap_vs_silicon_65280_10": round(float(np.mean(o15s)) - 65280.10, 2),
        "dl1_mean": round(float(np.mean(dl1s)), 2),
        "dl1_phase_sd": round(float(np.std(dl1s, ddof=1)), 2),
        "dl1_range": [min(dl1s), max(dl1s)],
        "dl2_mean": round(float(np.mean(dl2s)), 2),
        "dl2_phase_sd": round(float(np.std(dl2s, ddof=1)), 2),
        "dl2_range": [min(dl2s), max(dl2s)],
        "dtot_mean": round(float(np.mean(dtots)), 2),
        "dtot_phase_sd": round(float(np.std(dtots, ddof=1)), 2),
        "dtot_range": [min(dtots), max(dtots)],
        "noref_66runs_dl1_mean": round(float(np.mean(dl1_nr66)), 2),
        "noref_66runs_dl1_range": [min(dl1_nr66), max(dl1_nr66)],
    }

    return {"E1_noref_51phases": e1_res, "E3_exact_80_90_0ppm_66runs": e3_exact_80_90}


def run_e2_e5_e6() -> dict:
    d2 = json.loads((BUILD_DIR / "deep_rtl_validation_part2.json").read_text())
    b14 = json.loads((BUILD_DIR / "b1_b4_rtl_sim_results.json").read_text())
    wl_map = {s["name"]: make_synthetic_integer_ffn(s["d_model"], s["d_ff"], seed=20261130 + i) for i, s in enumerate(SHAPES_6)}

    # E2: Full 6-shape 60/60 table: 0 ppm vs +400 ppm vs silicon
    e60_recs = [r for r in d2["records"] if r["tag"] == "e60_s60"]
    sim_0ppm = next(x["sim_totals_6shapes"] for x in b14["B2_60_60_ppm_sweep"] if x["case"].startswith("0ppm"))
    sim_400ppm = next(x["sim_totals_6shapes"] for x in b14["B2_60_60_ppm_sweep"] if x["case"].startswith("+400ppm_meas"))
    e2_rows = []
    for r, s0, s400 in zip(e60_recs, sim_0ppm, sim_400ppm):
        e2_rows.append({
            "shape": r["shape"],
            "silicon_mean": r["meas_tot_mean"],
            "silicon_sd": r["meas_tot_sd"],
            "sim_0ppm": s0,
            "sim_0ppm_minus_sil": round(s0 - r["meas_tot_mean"], 2),
            "sim_400ppm": s400,
            "sim_400ppm_minus_sil": round(s400 - r["meas_tot_mean"], 2),
        })

    # E5: Residual regression AFTER B3 (+1.0 cyc/pkt at 1:1) on 60-pt set
    oos = [r for r in d2["records"] if r["group"] == "oos_60pt"]
    X_rows, y_post = [], []
    for r in oos:
        w = wl_map[r["shape"]]
        M_tot = w["l1_meta"]["n_tiles"] * w["l1_meta"]["k_chunks"] + w["l2_meta"]["n_tiles"] * w["l2_meta"]["k_chunks"]
        kch_max = max(w["l1_meta"]["k_chunks"], w["l2_meta"]["k_chunks"])
        is_1to1 = abs(r["eng_mhz"] - r["sdram_mhz"]) / float(r["sdram_mhz"]) < 0.01
        pred_post = r["wb_pred"] + (1.0 * M_tot if is_1to1 else 0.0)
        res_post = r["meas_tot_mean"] - pred_post  # silicon - model_post
        X_rows.append([1.0, float(M_tot), float(r["sdram_mhz"]), float(kch_max)])
        y_post.append(res_post)
    X = np.array(X_rows); y = np.array(y_post)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    r2 = 1.0 - float(np.sum((y - y_hat) ** 2)) / float(np.sum((y - np.mean(y)) ** 2))
    # Also split remaining bias by regime
    bw_res = [y[i] for i, r in enumerate(oos) if r["sdram_mhz"] <= 88]
    tr_res = [y[i] for i, r in enumerate(oos) if 90 <= r["sdram_mhz"] <= 96]
    cs_res = [y[i] for i, r in enumerate(oos) if r["sdram_mhz"] >= 100]

    # E6: Apply B3 (+1.0 cyc/pkt) to ALL 1:1 conditions in the 78-pt set (60/60 + 80/80)
    e6_rows = []
    for tag in ("e60_s60", "e80_s80"):
        sub = [r for r in d2["records"] if r["tag"] == tag]
        orig_e, post_e, orig_c, post_c = [], [], [], []
        for r in sub:
            w = wl_map[r["shape"]]
            M_tot = w["l1_meta"]["n_tiles"] * w["l1_meta"]["k_chunks"] + w["l2_meta"]["n_tiles"] * w["l2_meta"]["k_chunks"]
            p_o = r["wb_pred"]
            p_p = r["wb_pred"] + 1.0 * M_tot
            orig_e.append(abs(p_o - r["meas_tot_mean"]) / r["meas_tot_mean"] * 100.0)
            post_e.append(abs(p_p - r["meas_tot_mean"]) / r["meas_tot_mean"] * 100.0)
            orig_c.append(p_o - r["meas_tot_mean"])
            post_c.append(p_p - r["meas_tot_mean"])
        e6_rows.append({
            "tag": tag,
            "orig_mean_pct": round(float(np.mean(orig_e)), 5),
            "orig_max_pct": round(float(np.max(orig_e)), 5),
            "orig_mean_bias_cyc": round(float(np.mean(orig_c)), 2),
            "post_b3_mean_pct": round(float(np.mean(post_e)), 5),
            "post_b3_max_pct": round(float(np.max(post_e)), 5),
            "post_b3_mean_bias_cyc": round(float(np.mean(post_c)), 2),
        })

    return {
        "E2_60_60_6shapes": e2_rows,
        "E5_post_b3_residual_regression": {
            "features": ["intercept", "M_packets", "f_sdram_mhz", "k_chunks_max"],
            "beta": [round(float(b), 4) for b in beta],
            "r2": round(r2, 4),
            "mean_bias_by_regime_sil_minus_model_cyc": {
                "BW_bound_le88": round(float(np.mean(bw_res)), 2),
                "Transition_90_96": round(float(np.mean(tr_res)), 2),
                "Compute_sat_ge100": round(float(np.mean(cs_res)), 2),
            },
        },
        "E6_apply_b3_all_1to1": e6_rows,
    }


def main() -> int:
    res = {**run_e1_and_e3_offset_sims(), **run_e2_e5_e6()}
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
