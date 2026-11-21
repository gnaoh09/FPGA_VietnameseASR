#!/usr/bin/env python3
"""Exact 0-ppm same-VCO 3-way error decomposition across all 78 measured points (60 OOS + 18 CDC):
    (model - silicon) = (model - sim_0ppm) + (sim_0ppm - silicon)
Runs cycle-accurate RTL simulation (tb/tb_78pt_sim.v) using exact rational 0-ppm PLL clocks
(nominal eng_mhz / sdram_mhz) from build/deep_rtl_validation_part2.json, eliminating JTAG
read-window frequency noise. Uses unified signed residual convention: e = pred - meas.
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
IN_JSON = BUILD_DIR / "deep_rtl_validation_part2.json"
OUT_JSON = BUILD_DIR / "three_way_78pt_0ppm_decomp.json"


def summarize_subset(rows: list[dict], use_ext: bool = False) -> dict:
    if not rows:
        return {"n": 0}
    m_sim_pct = []
    s_sil_pct = []
    m_sil_pct = []
    m_sim_cyc = []
    s_sil_cyc = []
    m_sil_cyc = []
    for x in rows:
        wb_p = x["wb_ext_pred"] if use_ext else x["wb_pred"]
        sim_t = x["sim_0ppm_tot"]
        sil_t = x["silicon_mean"]
        ms = wb_p - sim_t
        ss = sim_t - sil_t
        msil = wb_p - sil_t
        m_sim_cyc.append(ms)
        s_sil_cyc.append(ss)
        m_sil_cyc.append(msil)
        m_sim_pct.append(abs(ms) / sim_t * 100.0)
        s_sil_pct.append(abs(ss) / sil_t * 100.0)
        m_sil_pct.append(abs(msil) / sil_t * 100.0)
    return {
        "n": len(rows),
        "abs_model_sim_mean_pct": round(float(np.mean(m_sim_pct)), 4),
        "abs_model_sim_max_pct": round(float(np.max(m_sim_pct)), 4),
        "signed_model_minus_sim_mean_cyc": round(float(np.mean(m_sim_cyc)), 2),
        "signed_model_minus_sim_range_cyc": [round(float(min(m_sim_cyc)), 2), round(float(max(m_sim_cyc)), 2)],
        "abs_sim_sil_mean_pct": round(float(np.mean(s_sil_pct)), 4),
        "abs_sim_sil_max_pct": round(float(np.max(s_sil_pct)), 4),
        "signed_sim_minus_sil_mean_cyc": round(float(np.mean(s_sil_cyc)), 2),
        "signed_sim_minus_sil_range_cyc": [round(float(min(s_sil_cyc)), 2), round(float(max(s_sil_cyc)), 2)],
        "abs_model_sil_mean_pct": round(float(np.mean(m_sil_pct)), 4),
        "abs_model_sil_max_pct": round(float(np.max(m_sil_pct)), 4),
        "signed_model_minus_sil_mean_cyc": round(float(np.mean(m_sil_cyc)), 2),
    }


def main() -> int:
    d = json.loads(IN_JSON.read_text(encoding="utf-8"))
    wl_map = {
        s["name"]: make_synthetic_integer_ffn(s["d_model"], s["d_ff"], seed=20261130 + i)
        for i, s in enumerate(SHAPES_6)
    }

    # Compile one vvp binary per nominal sdram_mhz (so sdram_ctrl CLK_MHZ / REF_INTERVAL / T_RFC match)
    vvps: dict[int, Path] = {}
    for sd_nom in sorted(set(int(r["sdram_mhz"]) for r in d["records"])):
        vvp = BUILD_DIR / f"tb_78pt_0ppm_sd{sd_nom}.vvp"
        subprocess.run(
            [
                "iverilog", "-g2012", "-DSIMULATION",
                f"-Ptb_78pt_sim.CLK_MHZ_P={sd_nom}",
                "-o", str(vvp),
                "rtl/sdram_ctrl.v", "rtl/zipformer_engine.v", "tb/tb_78pt_sim.v",
            ],
            cwd=str(ROOT_DIR),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        vvps[sd_nom] = vvp

    rows = []
    for r in d["records"]:
        eng_nom = float(r["eng_mhz"])
        sd_nom = float(r["sdram_mhz"])
        eng_half = 500.0 / eng_nom
        sd_half = 500.0 / sd_nom
        w = wl_map[r["shape"]]
        m1, m2 = w["l1_meta"], w["l2_meta"]

        cmd = [
            "vvp", str(vvps[int(r["sdram_mhz"])]),
            f"+ENG_HALF_NS={eng_half:.9f}",
            f"+SDRAM_HALF_NS={sd_half:.9f}",
            f"+L1_KC={m1['k_chunk']}", f"+L1_NT={m1['n_tiles']}", f"+L1_KCH={m1['k_chunks']}",
            f"+L1_WW={m1['w_words_per_chunk']}", f"+L1_BASE={r['l1_base']}",
            f"+L2_KC={m2['k_chunk']}", f"+L2_NT={m2['n_tiles']}", f"+L2_KCH={m2['k_chunks']}",
            f"+L2_WW={m2['w_words_per_chunk']}", f"+L2_BASE={r['l2_base']}",
            f"+DM={w['d_model']}", f"+DF={w['d_ff']}",
        ]
        out = subprocess.check_output(cmd, cwd=str(ROOT_DIR), text=True)
        m = re.search(r"SIM_RES l1_tot=(\d+) l1_st=(\d+) l2_tot=(\d+) l2_st=(\d+) tot=(\d+) st=(\d+)", out)
        assert m, f"Missing SIM_RES for {r['tag']} {r['shape']}: {out}"
        sim_tot = int(m.group(5))

        # Analytical WB prediction at exact 0-ppm nominal clocks
        p1 = predict_layer_param(r["l1_base"], m1, eng_nom, sd_nom, r["ref_interval"], 14.0, 3.0, 4.0, float(r["t_rfc_nom"]))
        p2 = predict_layer_param(r["l2_base"], m2, eng_nom, sd_nom, r["ref_interval"], 14.0, 3.0, 4.0, float(r["t_rfc_nom"]))
        wb_pred = p1 + p2
        m_tot = m1["n_tiles"] * m1["k_chunks"] + m2["n_tiles"] * m2["k_chunks"]
        is_1to1 = (int(r["eng_mhz"]) == int(r["sdram_mhz"]))
        wb_ext_pred = wb_pred + (float(m_tot) if is_1to1 else 0.0)

        sil_tot = float(r["meas_tot_mean"])
        bist_pred = float(r["bist_pred"])

        rows.append({
            "tag": r["tag"],
            "group": r["group"],
            "shape": r["shape"],
            "eng_mhz": int(r["eng_mhz"]),
            "sdram_mhz": int(r["sdram_mhz"]),
            "is_1to1": is_1to1,
            "m_tot": int(m_tot),
            "silicon_mean": sil_tot,
            "silicon_sd": float(r["meas_tot_sd"]),
            "sim_0ppm_tot": sim_tot,
            "wb_pred": round(wb_pred, 4),
            "wb_ext_pred": round(wb_ext_pred, 4),
            "bist_pred": round(bist_pred, 4),
            "model_minus_sim": round(wb_pred - sim_tot, 2),
            "ext_minus_sim": round(wb_ext_pred - sim_tot, 2),
            "sim_minus_silicon": round(sim_tot - sil_tot, 2),
            "model_minus_silicon": round(wb_pred - sil_tot, 2),
            "ext_minus_silicon": round(wb_ext_pred - sil_tot, 2),
            "abs_model_sim_pct": round(abs(wb_pred - sim_tot) / sim_tot * 100.0, 5),
            "abs_ext_sim_pct": round(abs(wb_ext_pred - sim_tot) / sim_tot * 100.0, 5),
            "abs_sim_sil_pct": round(abs(sim_tot - sil_tot) / sil_tot * 100.0, 5),
            "abs_model_sil_pct": round(abs(wb_pred - sil_tot) / sil_tot * 100.0, 5),
            "abs_ext_sil_pct": round(abs(wb_ext_pred - sil_tot) / sil_tot * 100.0, 5),
            "abs_bist_sil_pct": round(abs(bist_pred - sil_tot) / sil_tot * 100.0, 5),
        })

    oos = [x for x in rows if x["group"] == "oos_60pt"]
    subsets = {
        "all_78": summarize_subset(rows, use_ext=False),
        "oos_60": summarize_subset(oos, use_ext=False),
        "oos_54_non1to1": summarize_subset([x for x in oos if not x["is_1to1"]], use_ext=False),
        "oos_6_e80_s80": summarize_subset([x for x in oos if x["tag"] == "e80_s80"], use_ext=False),
        "oos_6_e80_s80_ext1to1": summarize_subset([x for x in oos if x["tag"] == "e80_s80"], use_ext=True),
        "cdc_6_e60_s60": summarize_subset([x for x in rows if x["tag"] == "e60_s60"], use_ext=False),
        "cdc_6_e60_s60_ext1to1": summarize_subset([x for x in rows if x["tag"] == "e60_s60"], use_ext=True),
        "cdc_6_e60_s90": summarize_subset([x for x in rows if x["tag"] == "e60_s90"], use_ext=False),
        "cdc_6_e70_s100": summarize_subset([x for x in rows if x["tag"] == "e70_s100"], use_ext=False),
    }

    regimes_oos = {
        "BW_bound_le88": summarize_subset([x for x in oos if x["sdram_mhz"] <= 88], use_ext=False),
        "Transition_90_96": summarize_subset([x for x in oos if 90 <= x["sdram_mhz"] <= 96], use_ext=False),
        "Compute_sat_ge100": summarize_subset([x for x in oos if x["sdram_mhz"] >= 100], use_ext=False),
    }

    per_freq_oos = []
    for sd_f in sorted(set(x["sdram_mhz"] for x in oos)):
        frows = [x for x in oos if x["sdram_mhz"] == sd_f]
        per_freq_oos.append({
            "sdram_mhz": sd_f,
            "wb_base_mean_pct": round(float(np.mean([x["abs_model_sil_pct"] for x in frows])), 5),
            "wb_ext_mean_pct": round(float(np.mean([x["abs_ext_sil_pct"] for x in frows])), 5),
            "sim_0ppm_sil_mean_pct": round(float(np.mean([x["abs_sim_sil_pct"] for x in frows])), 5),
            "bist_mean_pct": round(float(np.mean([x["abs_bist_sil_pct"] for x in frows])), 5),
        })

    payload = {
        "convention": "signed_residual = pred - meas (negative = underprediction)",
        "subsets": subsets,
        "regimes_oos": regimes_oos,
        "per_freq_oos": per_freq_oos,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=== SUBSETS ===")
    print(json.dumps(subsets, indent=2))
    print("=== REGIMES (OOS 60) ===")
    print(json.dumps(regimes_oos, indent=2))
    print("=== PER-FREQ OOS ===")
    for pf in per_freq_oos:
        print(" ", pf)
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
