#!/usr/bin/env python3
"""Run silicon measurements for frozen experiments C1..C4 on EP4CE10E22C8N
and compare against build/frozen_heldout_predictions_c1_c4.json (locked at commit 4bbade7).

Always restores build/output_files/top.svf in finally:.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpga_jtag_runner import OP_COMMIT_ACT, UsbBlasterJtag, pack_cfg1, pack_cfg2
from gen_golden_vectors import pack_words_128bit_i8
from run_43_silicon_heldout import make_ffn
from run_deep_rtl_validation_part1 import run_ffn_pair, setup_shape
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks, run_bist_stats
from sdram_helpers import run_sdram_stream_layer, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = BUILD_DIR / "output_files"
SVF_DEFAULT = OF_DIR / "top.svf"
FROZEN_JSON = BUILD_DIR / "frozen_heldout_predictions_c1_c4.json"
OUT_JSON = BUILD_DIR / "c1_c4_silicon_results.json"


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def run_wl_list(dev: UsbBlasterJtag, wls: list[dict], t_tile: int, n_rep: int = 10) -> list[dict]:
    out = []
    for w in wls:
        x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
        hpost_base = len(x_words)
        for i, xw in enumerate(x_words):
            dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
        dev.flush()
        write_lut_256(dev, w["lut"], lut_bank=0)
        tots, stalls = [], []
        for rep in range(n_rep):
            c1 = run_sdram_stream_layer(
                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                t_tile, w["d_model"], w["d_ff"], w["l1_meta"], w["l1_base"],
                4, 1, 0, hpost_base, 1
            )
            c2 = run_sdram_stream_layer(
                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                t_tile, w["d_ff"], w["d_model"], w["l2_meta"], w["l2_base"],
                4, 0, hpost_base, 0, 0
            )
            if rep == 0:
                y_exp = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]
                assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"0-LSB fail {w['shape']}"
            tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
            stalls.append(int(c1["stall_cycles"] + c2["stall_cycles"]))
        meas_m = float(np.mean(tots))
        meas_sd = float(np.std(tots, ddof=1))
        err_pct = abs(w["pred_total_cyc"] - meas_m) / meas_m * 100.0
        out.append({
            **{k: w[k] for k in w if k in ("shape", "t_tile", "eng_mhz", "sdram_mhz", "l1_kchunks", "l2_kchunks", "pred_total_cyc")},
            "meas_mean": round(meas_m, 2),
            "meas_sd": round(meas_sd, 2),
            "stall_pct": round(100.0 * float(np.mean(stalls)) / meas_m, 2),
            "err_pct": round(err_pct, 5),
        })
    return out


def main() -> int:
    frozen = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    results: dict = {"frozen_commit": "4bbade777099f4bcbf7602b79f134104c3d105b6"}

    try:
        # C1: Held-out extrapolation (T_tile=4 @ 80/100 MHz)
        c1_wls, c1_words = [], []
        for idx, p in enumerate(frozen["c1_heldout_extrapolation"]):
            wl = make_ffn(p["d_model"], p["d_ff"], 4, seed=20261201 + idx)
            assert len(c1_words) == p["l1_base"]
            c1_words.extend(wl["l1_words"])
            assert len(c1_words) == p["l2_base"]
            c1_words.extend(wl["l2_words"])
            c1_wls.append({**wl, **p})
        flash_svf(SVF_DEFAULT)
        dev = UsbBlasterJtag()
        try:
            eng_m, sd_m, _, _ = measure_pll_clocks(dev)
            dev.sdram_write_bulk_words(0, c1_words)
            c1_res = run_wl_list(dev, c1_wls, 4, n_rep=10)
            results["c1_heldout"] = {"eng_meas": eng_m, "sd_meas": sd_m, "rows": c1_res}
            print("[C1]", json.dumps(c1_res))
        finally:
            dev.close()

        # C2: T_tile=8 held-out @ {65, 75, 110} MHz (+ C4 reference @ 110 MHz)
        t8_wls_base, t8_words = [], []
        c2_specs = [(192, 384), (256, 576), (256, 960), (512, 1536)]
        for idx, (dm, df) in enumerate(c2_specs):
            wl = make_ffn(dm, df, 8, seed=20261210 + idx)
            b1 = len(t8_words); t8_words.extend(wl["l1_words"])
            b2 = len(t8_words); t8_words.extend(wl["l2_words"])
            t8_wls_base.append({**wl, "shape": f"{dm}->{df}->{dm}", "l1_base": b1, "l2_base": b2})

        c2_rows = []
        for sd_mhz in (65, 75, 110):
            flash_svf(OF_DIR / f"top_t8_sd{sd_mhz}.svf")
            dev = UsbBlasterJtag()
            try:
                eng_m, sd_m, _, _ = measure_pll_clocks(dev)
                bist = run_bist_stats(dev, 65536, sd_m)
                assert bist["err_cnt"] == 0
                dev.sdram_write_bulk_words(0, t8_words)
                preds_m = {x["shape"]: x for x in frozen["c2_t8_new_clocks"] if x["sdram_mhz"] == sd_mhz}
                wls_here = [{**w, **preds_m[w["shape"]]} for w in t8_wls_base]
                rows = run_wl_list(dev, wls_here, 8, n_rep=10)
                for r in rows:
                    r["eng_meas"] = eng_m
                    r["sd_meas"] = sd_m
                    r["bist_rd_MBps"] = bist["rd_MBps_1e6"]
                c2_rows.extend(rows)
                print(f"[C2 @ {sd_mhz}M]", json.dumps(rows))
            finally:
                dev.close()
        results["c2_t8_new_clocks"] = c2_rows

        # C3: Offset sweep 0..31 (n=10) on 192->512->192 @ 90 MHz
        wl_512 = make_ffn(192, 512, 4, seed=20261220)
        flash_svf(OF_DIR / "top_sdram90.svf")
        dev = UsbBlasterJtag()
        try:
            eng90, sd90, _, _ = measure_pll_clocks(dev)
            setup_shape(dev, wl_512)
            dev.sdram_write_bulk_words(16384, wl_512["l1_words"])
            dev.sdram_write_bulk_words(32768, wl_512["l2_words"])
            c3_rows = []
            for p in frozen["c3_offset_192_512_90mhz"]:
                runs = [run_ffn_pair(dev, wl_512, p["l1_base"], p["l2_base"]) for _ in range(10)]
                tots = [r["total_cycles"] for r in runs]
                c3_rows.append({
                    "offset": p["offset"],
                    "pred_total_cyc": p["pred_total_cyc"],
                    "meas_mean": round(float(np.mean(tots)), 2),
                    "meas_sd": round(float(np.std(tots, ddof=1)), 2),
                })
            preds = np.array([x["pred_total_cyc"] for x in c3_rows])
            meass = np.array([x["meas_mean"] for x in c3_rows])
            sds = np.array([x["meas_sd"] for x in c3_rows])
            slope, intercept = np.polyfit(preds, meass, 1)
            mae = float(np.mean(np.abs(meass - preds)))
            results["c3_offset_192_512_90mhz"] = {
                "eng_meas": eng90, "sd_meas": sd90,
                "slope": round(float(slope), 4),
                "intercept": round(float(intercept), 2),
                "mae_cyc": round(mae, 2),
                "mean_sd_cyc": round(float(np.mean(sds)), 2),
                "rows": c3_rows,
            }
            print(f"[C3 192->512 0..31 n=10 @ 90M] slope={slope:.4f}, MAE={mae:.2f}, mean_SD={np.mean(sds):.2f}")
        finally:
            dev.close()

        # C4: Direct silicon verification of WB choice (80 MHz) vs BIST choice (50 MHz) vs 110 MHz ref on T_tile=8 192->384->192
        # Note: 50, 80, 110 MHz T_tile=8 measurements are already recorded in part3 (50, 80) and C2 (110) with exact same shape!
        r110 = next(x for x in c2_rows if x["sdram_mhz"] == 110 and x["shape"] == "192->384->192")
        d3 = json.loads((BUILD_DIR / "deep_rtl_validation_part3.json").read_text())
        r80 = next(x for x in d3["t8_freq_sweep"] if x["mhz"] == 80 and x["shape"] == "ffn_192_384_192")
        r50 = next(x for x in d3["t8_freq_sweep"] if x["mhz"] == 50 and x["shape"] == "ffn_192_384_192")
        results["c4_dse_verification"] = {
            "ref_110mhz_meas": r110["meas_mean"],
            "wb_choice_80mhz_meas": r80["meas_tot_mean"],
            "wb_penalty_vs_110_pct": round((r80["meas_tot_mean"] - r110["meas_mean"]) / r110["meas_mean"] * 100.0, 2),
            "bist_choice_50mhz_meas": r50["meas_tot_mean"],
            "bist_penalty_vs_110_pct": round((r50["meas_tot_mean"] - r110["meas_mean"]) / r110["meas_mean"] * 100.0, 2),
        }
        print("[C4 DSE]", json.dumps(results["c4_dse_verification"]))

        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved: {OUT_JSON}")
    finally:
        if SVF_DEFAULT.exists():
            flash_svf(SVF_DEFAULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
