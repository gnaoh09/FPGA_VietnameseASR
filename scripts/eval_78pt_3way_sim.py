#!/usr/bin/env python3
"""Three-way error decomposition across all 78 measured points (60 OOS + 18 CDC):
    total_err = (model - sim) + (sim - silicon)
Runs cycle-accurate RTL simulation (tb/tb_78pt_sim.v) using the exact measured PLL clocks
(meas_eng_mhz, meas_sdram_mhz) from build/deep_rtl_validation_part2.json.
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
OUT_JSON = BUILD_DIR / "three_way_78pt_sim_decomp.json"


def main() -> int:
    d = json.loads(IN_JSON.read_text(encoding="utf-8"))
    fmeta = {m["tag"]: m for m in d["freq_meta"]}
    wl_map = {s["name"]: make_synthetic_integer_ffn(s["d_model"], s["d_ff"], seed=20261130 + i) for i, s in enumerate(SHAPES_6)}

    # Compile one vvp binary per nominal sdram_mhz (so sdram_ctrl CLK_MHZ / REF_INTERVAL / T_RFC match)
    vvps: dict[int, Path] = {}
    for sd_nom in sorted(set(r["sdram_mhz"] for r in d["records"])):
        vvp = BUILD_DIR / f"tb_78pt_sd{sd_nom}.vvp"
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
        fm = fmeta[r["tag"]]
        eng_meas = fm["meas_eng_mhz"]
        sd_meas = fm["meas_sdram_mhz"]
        eng_half = 500.0 / eng_meas
        sd_half = 500.0 / sd_meas
        w = wl_map[r["shape"]]
        m1, m2 = w["l1_meta"], w["l2_meta"]

        cmd = [
            "vvp", str(vvps[r["sdram_mhz"]]),
            f"+ENG_HALF_NS={eng_half:.6f}",
            f"+SDRAM_HALF_NS={sd_half:.6f}",
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

        # Analytical WB prediction using the EXACT same measured clocks
        p1 = predict_layer_param(r["l1_base"], m1, eng_meas, sd_meas, r["ref_interval"], 14.0, 3.0, 4.0, float(r["t_rfc_nom"]))
        p2 = predict_layer_param(r["l2_base"], m2, eng_meas, sd_meas, r["ref_interval"], 14.0, 3.0, 4.0, float(r["t_rfc_nom"]))
        mod_meas = p1 + p2
        sil_tot = r["meas_tot_mean"]

        rows.append({
            "tag": r["tag"],
            "group": r["group"],
            "shape": r["shape"],
            "eng_meas": eng_meas,
            "sd_meas": sd_meas,
            "silicon_mean": sil_tot,
            "silicon_sd": r["meas_tot_sd"],
            "sim_tot": sim_tot,
            "model_meas_clk": round(mod_meas, 2),
            "model_minus_sim": round(mod_meas - sim_tot, 2),
            "sim_minus_silicon": round(sim_tot - sil_tot, 2),
            "model_minus_silicon": round(mod_meas - sil_tot, 2),
            "abs_model_sim_pct": round(abs(mod_meas - sim_tot) / sim_tot * 100.0, 5),
            "abs_sim_sil_pct": round(abs(sim_tot - sil_tot) / sil_tot * 100.0, 5),
        })

    oos = [x for x in rows if x["group"] == "oos_60pt"]
    summary = {
        "n_all": len(rows),
        "n_oos": len(oos),
        "oos_model_minus_sim_cyc_mean": round(float(np.mean([x["model_minus_sim"] for x in oos])), 2),
        "oos_model_minus_sim_pct_mean": round(float(np.mean([x["abs_model_sim_pct"] for x in oos])), 5),
        "oos_sim_minus_silicon_cyc_mean": round(float(np.mean([x["sim_minus_silicon"] for x in oos])), 2),
        "oos_sim_minus_silicon_pct_mean": round(float(np.mean([x["abs_sim_sil_pct"] for x in oos])), 5),
        "oos_sim_minus_silicon_cyc_range": [round(min(x["sim_minus_silicon"] for x in oos), 2), round(max(x["sim_minus_silicon"] for x in oos), 2)],
        "e80_s80_rows": [x for x in oos if x["tag"] == "e80_s80"],
    }
    print(json.dumps(summary, indent=2))
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
