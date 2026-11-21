#!/usr/bin/env python3
"""Run B1 (pre-locked held-out test set), B5.3/B5.4 (extended offset on silicon),
and B6.2/B6.3 (silicon DSE packing validation) on EP4CE10E22C8N.

1. Computes SHA-256 hashes of RTL + model scripts, freezes white-box and
   grey-box parameters, and saves pre-registered predictions to
   build/b1_preregistered_heldout.json BEFORE flashing the FPGA.
2. Executes all silicon measurements with n=10 repeats and 0-LSB checks.
3. Always restores build/output_files/top.svf in finally:.
4. Saves full results to build/b1_b5_b6_silicon_results.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpga_jtag_runner import OP_COMMIT_ACT, UsbBlasterJtag, pack_cfg1, pack_cfg2
from gen_golden_vectors import pack_words_128bit_i8
from run_deep_rtl_validation_part2 import (
    predict_bist_baseline,
    predict_layer_param,
)
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks, run_bist_stats
from run_oos_stall_validation import make_synthetic_integer_ffn
from run_t_tile8_ep4ce10 import make_ffn_t8, predict_layer_rtl_t8
from sdram_helpers import run_sdram_stream_layer, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = BUILD_DIR / "output_files"
SVF_DEFAULT = OF_DIR / "top.svf"
PART2_JSON = BUILD_DIR / "deep_rtl_validation_part2.json"
PREREG_JSON = BUILD_DIR / "b1_preregistered_heldout.json"
OUT_JSON = BUILD_DIR / "b1_b5_b6_silicon_results.json"

N_REPEATS = 10
RD_LAT = 4

B1_SHAPES = [
    {"name": "ffn_192_448_192", "d_model": 192, "d_ff": 448},
    {"name": "ffn_256_640_256", "d_model": 256, "d_ff": 640},
    {"name": "ffn_384_1024_384", "d_model": 384, "d_ff": 1024},
]

B1_T4_BITSTREAMS = [
    {"tag": "e60_s90", "eng_mhz": 60, "sdram_mhz": 90, "svf": OF_DIR / "top_eng60_sd90.svf"},
    {"tag": "e70_s100", "eng_mhz": 70, "sdram_mhz": 100, "svf": OF_DIR / "top_eng70_sd100.svf"},
    {"tag": "e90_s100", "eng_mhz": 90, "sdram_mhz": 100, "svf": OF_DIR / "top_eng90_sd100.svf"},
]

B1_T8_BITSTREAMS = [
    {"tag": "t8_e80_s40", "eng_mhz": 80, "sdram_mhz": 40, "svf": OF_DIR / "top_t8_sd40.svf"},
    {"tag": "t8_e80_s50", "eng_mhz": 80, "sdram_mhz": 50, "svf": OF_DIR / "top_t8_sd50.svf"},
    {"tag": "t8_e80_s100", "eng_mhz": 80, "sdram_mhz": 100, "svf": OF_DIR / "top_t8.svf"},
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def compute_active_cycles(layer_meta: dict, t_tile: int) -> int:
    n_tiles = layer_meta["n_tiles"]
    k_chunks = layer_meta["k_chunks"]
    k_chunk = layer_meta["k_chunk"]
    per_tile = sum(t_tile * k_chunk + (13 if kc == k_chunks - 1 else 6) for kc in range(k_chunks))
    return n_tiles * per_tile


def load_frozen_grey_params() -> dict:
    d = json.loads(PART2_JSON.read_text())
    folds = d["summary"]["grey_loso_folds"]
    return {
        "c_page": float(np.mean([f["fitted_c_page"] for f in folds])),
        "c_cdc_sd": float(np.mean([f["fitted_c_cdc_sd"] for f in folds])),
        "c_cdc_eng": float(np.mean([f["fitted_c_cdc_eng"] for f in folds])),
        "t_rfc": float(np.mean([f["fitted_t_rfc"] for f in folds])),
        "source": "mean of 6 LOSO folds in build/deep_rtl_validation_part2.json",
    }


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    grey_params = load_frozen_grey_params()

    # Prepare B1 T=4 workloads (3 new shapes)
    t4_wls = []
    t4_sdram_words: list[int] = []
    for idx, spec in enumerate(B1_SHAPES):
        wl = make_synthetic_integer_ffn(spec["d_model"], spec["d_ff"], seed=20261200 + idx)
        l1_base = len(t4_sdram_words)
        t4_sdram_words.extend(wl["l1_words"])
        l2_base = len(t4_sdram_words)
        t4_sdram_words.extend(wl["l2_words"])
        t4_wls.append({**spec, **wl, "l1_base": l1_base, "l2_base": l2_base})

    # Prepare B1 T=8 workloads (3 new shapes)
    t8_wls = []
    t8_sdram_words: list[int] = []
    for idx, spec in enumerate(B1_SHAPES):
        wl = make_ffn_t8(spec["d_model"], spec["d_ff"], seed=20261210 + idx)
        l1_base = len(t8_sdram_words)
        t8_sdram_words.extend(wl["l1_words"])
        l2_base = len(t8_sdram_words)
        t8_sdram_words.extend(wl["l2_words"])
        t8_wls.append({**spec, **wl, "l1_base": l1_base, "l2_base": l2_base})

    # Prepare B5.3 (T=8, 192->640->192, offset 1 vs 15) and B5.4 (T=4, 192->640->192, 0..31)
    wl_192_640_t8 = make_ffn_t8(192, 640, seed=20261220)
    wl_192_640_t4 = make_synthetic_integer_ffn(192, 640, seed=20261221)

    # Pre-register & freeze predictions BEFORE touching the FPGA
    prereg_t4 = []
    for bs in B1_T4_BITSTREAMS:
        ref_int = bs["sdram_mhz"] * 14
        t_rfc_nom = 9.0 if bs["sdram_mhz"] > 100 else 8.0
        for w in t4_wls:
            wb1 = predict_layer_param(w["l1_base"], w["l1_meta"], bs["eng_mhz"], bs["sdram_mhz"], ref_int, 14.0, 3.0, 4.0, t_rfc_nom)
            wb2 = predict_layer_param(w["l2_base"], w["l2_meta"], bs["eng_mhz"], bs["sdram_mhz"], ref_int, 14.0, 3.0, 4.0, t_rfc_nom)
            gb1 = predict_layer_param(
                w["l1_base"], w["l1_meta"], bs["eng_mhz"], bs["sdram_mhz"], ref_int,
                grey_params["c_page"], grey_params["c_cdc_sd"], grey_params["c_cdc_eng"], grey_params["t_rfc"],
            )
            gb2 = predict_layer_param(
                w["l2_base"], w["l2_meta"], bs["eng_mhz"], bs["sdram_mhz"], ref_int,
                grey_params["c_page"], grey_params["c_cdc_sd"], grey_params["c_cdc_eng"], grey_params["t_rfc"],
            )
            act = compute_active_cycles(w["l1_meta"], 4) + compute_active_cycles(w["l2_meta"], 4)
            prereg_t4.append({
                "tag": bs["tag"],
                "shape": w["name"],
                "eng_mhz": bs["eng_mhz"],
                "sdram_mhz": bs["sdram_mhz"],
                "l1_base": w["l1_base"],
                "l2_base": w["l2_base"],
                "active_cyc": act,
                "wb_pred_tot": round(wb1 + wb2, 3),
                "wb_pred_stall": round((wb1 + wb2) - act, 3),
                "gb_pred_tot": round(gb1 + gb2, 3),
                "gb_pred_stall": round((gb1 + gb2) - act, 3),
            })

    prereg_t8 = []
    for bs in B1_T8_BITSTREAMS:
        ref_int = bs["sdram_mhz"] * 14
        for w in t8_wls:
            p1 = predict_layer_rtl_t8(w["l1_base"], w["l1_meta"], float(bs["sdram_mhz"]), ref_int, 8, float(bs["eng_mhz"]))
            p2 = predict_layer_rtl_t8(w["l2_base"], w["l2_meta"], float(bs["sdram_mhz"]), ref_int, 8, float(bs["eng_mhz"]))
            wb_tot = p1["pred_total_cyc"] + p2["pred_total_cyc"]
            wb_st = p1["pred_stall_cyc"] + p2["pred_stall_cyc"]
            prereg_t8.append({
                "tag": bs["tag"],
                "shape": w["name"],
                "eng_mhz": bs["eng_mhz"],
                "sdram_mhz": bs["sdram_mhz"],
                "l1_base": w["l1_base"],
                "l2_base": w["l2_base"],
                "active_cyc": int(wb_tot - wb_st),
                "wb_pred_tot": round(wb_tot, 3),
                "wb_pred_stall": round(wb_st, 3),
            })

    prereg_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_hashes_sha256": {
            "rtl/zipformer_engine.v": sha256_file(ROOT_DIR / "rtl" / "zipformer_engine.v"),
            "rtl/sdram_ctrl.v": sha256_file(ROOT_DIR / "rtl" / "sdram_ctrl.v"),
            "scripts/run_oos_stall_validation.py": sha256_file(ROOT_DIR / "scripts" / "run_oos_stall_validation.py"),
            "scripts/run_deep_rtl_validation_part2.py": sha256_file(ROOT_DIR / "scripts" / "run_deep_rtl_validation_part2.py"),
            "scripts/run_t_tile8_ep4ce10.py": sha256_file(ROOT_DIR / "scripts" / "run_t_tile8_ep4ce10.py"),
        },
        "frozen_whitebox_constants": {"c_page": 14, "c_cdc_sd": 3, "c_cdc_eng": 4, "t_rfc_le100": 8},
        "frozen_greybox_constants": grey_params,
        "b1_t4_predictions": prereg_t4,
        "b1_t8_predictions": prereg_t8,
    }
    PREREG_JSON.write_text(json.dumps(prereg_payload, indent=2))
    print(f"[Pre-Reg] Frozen predictions written to {PREREG_JSON} at {prereg_payload['timestamp_utc']}")

    results: dict = {
        "device": "EP4CE10E22C8N",
        "prereg_timestamp_utc": prereg_payload["timestamp_utc"],
        "code_hashes_sha256": prereg_payload["code_hashes_sha256"],
        "frozen_greybox_constants": grey_params,
    }

    try:
        # ------------------------------------------------------------------
        # B1-T4: 3 unseen shapes x 3 unseen (eng, sdram) clock domains
        # ------------------------------------------------------------------
        b1_t4_records = []
        prereg_t4_map = {(r["tag"], r["shape"]): r for r in prereg_t4}
        for bs in B1_T4_BITSTREAMS:
            assert bs["svf"].exists(), f"Missing {bs['svf']}"
            flash_svf(bs["svf"])
            dev = UsbBlasterJtag()
            try:
                eng_m, sd_m, _, _ = measure_pll_clocks(dev)
                bist = run_bist_stats(dev, 65536, sd_m)
                assert bist["err_cnt"] == 0, f"BIST fail {bs['tag']}"
                dev.sdram_write_bulk_words(0, t4_sdram_words)
                print(f"[B1-T4 {bs['tag']}] eng={eng_m:.3f}M, sdram={sd_m:.3f}M, BIST={bist['rd_MBps_1e6']} MB/s")

                for w in t4_wls:
                    x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
                    hpost_base = len(x_words)
                    for i, xw in enumerate(x_words):
                        dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
                    dev.flush()
                    write_lut_256(dev, w["lut"], lut_bank=0)

                    tots, stalls = [], []
                    for rep in range(N_REPEATS):
                        c1 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            4, w["d_model"], w["d_ff"], w["l1_meta"], w["l1_base"],
                            RD_LAT, 1, 0, hpost_base, 1,
                        )
                        c2 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            4, w["d_ff"], w["d_model"], w["l2_meta"], w["l2_base"],
                            RD_LAT, 0, hpost_base, 0, 0,
                        )
                        if rep == 0:
                            y_exp = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]
                            assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"0-LSB fail {bs['tag']} {w['name']}"
                        tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                        stalls.append(int(c1["stall_cycles"] + c2["stall_cycles"]))

                    pr = prereg_t4_map[(bs["tag"], w["name"])]
                    bist_pred = predict_bist_baseline(w, float(bs["eng_mhz"]), bist["rd_MBps_1e6"])
                    meas_tot = float(np.mean(tots))
                    meas_tot_sd = float(np.std(tots, ddof=1))
                    meas_st = float(np.mean(stalls))
                    meas_st_sd = float(np.std(stalls, ddof=1))
                    b1_t4_records.append({
                        **pr,
                        "meas_eng_mhz": eng_m,
                        "meas_sdram_mhz": sd_m,
                        "meas_tot_mean": round(meas_tot, 2),
                        "meas_tot_sd": round(meas_tot_sd, 2),
                        "meas_stall_mean": round(meas_st, 2),
                        "meas_stall_sd": round(meas_st_sd, 2),
                        "wb_tot_err_pct": round(abs(pr["wb_pred_tot"] - meas_tot) / meas_tot * 100.0, 5),
                        "wb_stall_err_pct": round(abs(pr["wb_pred_stall"] - meas_st) / meas_st * 100.0, 5),
                        "gb_tot_err_pct": round(abs(pr["gb_pred_tot"] - meas_tot) / meas_tot * 100.0, 5),
                        "gb_stall_err_pct": round(abs(pr["gb_pred_stall"] - meas_st) / meas_st * 100.0, 5),
                        "bist_pred_tot": round(bist_pred, 2),
                        "bist_tot_err_pct": round(abs(bist_pred - meas_tot) / meas_tot * 100.0, 5),
                    })
                    print(f"  {w['name']}: meas={meas_tot:.1f}±{meas_tot_sd:.1f}, wb={pr['wb_pred_tot']:.1f} ({b1_t4_records[-1]['wb_tot_err_pct']:.4f}%), gb={pr['gb_pred_tot']:.1f} ({b1_t4_records[-1]['gb_tot_err_pct']:.4f}%)")
            finally:
                dev.close()
        results["b1_t4_records"] = b1_t4_records

        # ------------------------------------------------------------------
        # B1-T8 & B5.3: 3 new shapes @ T_tile=8 + offset 1 vs 15 @ T_tile=8
        # ------------------------------------------------------------------
        b1_t8_records = []
        b5_3_records = []
        prereg_t8_map = {(r["tag"], r["shape"]): r for r in prereg_t8}
        for bs in B1_T8_BITSTREAMS:
            assert bs["svf"].exists(), f"Missing {bs['svf']}"
            flash_svf(bs["svf"])
            dev = UsbBlasterJtag()
            try:
                eng_m, sd_m, _, _ = measure_pll_clocks(dev)
                bist = run_bist_stats(dev, 65536, sd_m)
                assert bist["err_cnt"] == 0, f"BIST fail {bs['tag']}"
                dev.sdram_write_bulk_words(0, t8_sdram_words)
                print(f"[B1-T8 {bs['tag']}] eng={eng_m:.3f}M, sdram={sd_m:.3f}M")

                for w in t8_wls:
                    x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
                    hpost_base = len(x_words)
                    for i, xw in enumerate(x_words):
                        dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
                    dev.flush()
                    write_lut_256(dev, w["lut"], lut_bank=0)

                    tots, stalls = [], []
                    for rep in range(N_REPEATS):
                        c1 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            8, w["d_model"], w["d_ff"], w["l1_meta"], w["l1_base"],
                            RD_LAT, 1, 0, hpost_base, 1,
                        )
                        c2 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            8, w["d_ff"], w["d_model"], w["l2_meta"], w["l2_base"],
                            RD_LAT, 0, hpost_base, 0, 0,
                        )
                        if rep == 0:
                            y_exp = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]
                            assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"T8 0-LSB fail {bs['tag']} {w['name']}"
                        tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                        stalls.append(int(c1["stall_cycles"] + c2["stall_cycles"]))

                    pr = prereg_t8_map[(bs["tag"], w["name"])]
                    meas_tot = float(np.mean(tots))
                    meas_tot_sd = float(np.std(tots, ddof=1))
                    meas_st = float(np.mean(stalls))
                    b1_t8_records.append({
                        **pr,
                        "meas_eng_mhz": eng_m,
                        "meas_sdram_mhz": sd_m,
                        "meas_tot_mean": round(meas_tot, 2),
                        "meas_tot_sd": round(meas_tot_sd, 2),
                        "meas_stall_mean": round(meas_st, 2),
                        "meas_stall_sd": round(float(np.std(stalls, ddof=1)), 2),
                        "wb_tot_err_pct": round(abs(pr["wb_pred_tot"] - meas_tot) / meas_tot * 100.0, 5),
                        "wb_stall_err_pct": round(abs(pr["wb_pred_stall"] - meas_st) / meas_st * 100.0, 5),
                    })
                    print(f"  T8 {w['name']}: meas={meas_tot:.1f}±{meas_tot_sd:.1f}, wb={pr['wb_pred_tot']:.1f} ({b1_t8_records[-1]['wb_tot_err_pct']:.4f}%)")

                # B5.3: T=8 offset 1 vs 15 on 192->640->192 at 50 MHz and 100 MHz
                if bs["sdram_mhz"] in (50, 100):
                    x_words = [int(s, 16) for s in pack_words_128bit_i8(wl_192_640_t8["x_q"])]
                    hpost_base = len(x_words)
                    for i, xw in enumerate(x_words):
                        dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
                    dev.flush()
                    write_lut_256(dev, wl_192_640_t8["lut"], lut_bank=0)

                    off_stats = {}
                    ref_int = bs["sdram_mhz"] * 14
                    for off in (1, 15):
                        b1 = 32 + off
                        b2 = ((b1 + len(wl_192_640_t8["l1_words"]) + 31) // 32) * 32 + off
                        dev.sdram_write_bulk_words(b1, wl_192_640_t8["l1_words"])
                        dev.sdram_write_bulk_words(b2, wl_192_640_t8["l2_words"])
                        tots, l1_st, l2_st = [], [], []
                        for _ in range(N_REPEATS):
                            c1 = run_sdram_stream_layer(
                                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                                8, 192, 640, wl_192_640_t8["l1_meta"], b1, RD_LAT, 1, 0, hpost_base, 1,
                            )
                            c2 = run_sdram_stream_layer(
                                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                                8, 640, 192, wl_192_640_t8["l2_meta"], b2, RD_LAT, 0, hpost_base, 0, 0,
                            )
                            tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                            l1_st.append(int(c1["stall_cycles"]))
                            l2_st.append(int(c2["stall_cycles"]))
                        p1 = predict_layer_rtl_t8(b1, wl_192_640_t8["l1_meta"], float(bs["sdram_mhz"]), ref_int, 8, float(bs["eng_mhz"]))
                        p2 = predict_layer_rtl_t8(b2, wl_192_640_t8["l2_meta"], float(bs["sdram_mhz"]), ref_int, 8, float(bs["eng_mhz"]))
                        off_stats[off] = {
                            "meas_tot_mean": round(float(np.mean(tots)), 2),
                            "meas_tot_sd": round(float(np.std(tots, ddof=1)), 2),
                            "meas_l1_stall": round(float(np.mean(l1_st)), 2),
                            "meas_l2_stall": round(float(np.mean(l2_st)), 2),
                            "pred_tot": round(p1["pred_total_cyc"] + p2["pred_total_cyc"], 2),
                        }
                    b5_3_records.append({
                        "tag": bs["tag"],
                        "sdram_mhz": bs["sdram_mhz"],
                        "off1": off_stats[1],
                        "off15": off_stats[15],
                        "meas_delta_1_15": round(off_stats[1]["meas_tot_mean"] - off_stats[15]["meas_tot_mean"], 2),
                        "pred_delta_1_15": round(off_stats[1]["pred_tot"] - off_stats[15]["pred_tot"], 2),
                    })
                    print(f"  [B5.3 @ {bs['sdram_mhz']}M T8] Delta(1-15) meas={b5_3_records[-1]['meas_delta_1_15']}, pred={b5_3_records[-1]['pred_delta_1_15']}")
            finally:
                dev.close()
        results["b1_t8_records"] = b1_t8_records
        results["b5_3_t8_offset"] = b5_3_records

        # ------------------------------------------------------------------
        # B5.4 (T=4 32-offset 0..31 @ 90 MHz) & B6.2/B6.3 (Silicon DSE @ 90 MHz)
        # ------------------------------------------------------------------
        flash_svf(OF_DIR / "top_sdram90.svf")
        dev = UsbBlasterJtag()
        try:
            eng_m, sd_m, _, _ = measure_pll_clocks(dev)
            bist = run_bist_stats(dev, 65536, sd_m)
            assert bist["err_cnt"] == 0, "BIST fail 90M"
            print(f"[90M] eng={eng_m:.3f}M, sdram={sd_m:.3f}M")

            # B5.4: 32-offset sweep 0..31 on 192->640->192 (n=10)
            x_words = [int(s, 16) for s in pack_words_128bit_i8(wl_192_640_t4["x_q"])]
            hpost_base = len(x_words)
            for i, xw in enumerate(x_words):
                dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
            dev.flush()
            write_lut_256(dev, wl_192_640_t4["lut"], lut_bank=0)

            off_sweep = []
            ref_int = 90 * 14
            for off in range(32):
                b1 = 32 + off
                b2 = ((b1 + len(wl_192_640_t4["l1_words"]) + 31) // 32) * 32 + off
                dev.sdram_write_bulk_words(b1, wl_192_640_t4["l1_words"])
                dev.sdram_write_bulk_words(b2, wl_192_640_t4["l2_words"])
                tots, l1_st, l2_st = [], [], []
                for _ in range(N_REPEATS):
                    c1 = run_sdram_stream_layer(
                        dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                        4, 192, 640, wl_192_640_t4["l1_meta"], b1, RD_LAT, 1, 0, hpost_base, 1,
                    )
                    c2 = run_sdram_stream_layer(
                        dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                        4, 640, 192, wl_192_640_t4["l2_meta"], b2, RD_LAT, 0, hpost_base, 0, 0,
                    )
                    tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                    l1_st.append(int(c1["stall_cycles"]))
                    l2_st.append(int(c2["stall_cycles"]))
                p1 = predict_layer_param(b1, wl_192_640_t4["l1_meta"], 80.0, 90.0, ref_int, 14.0, 3.0, 4.0, 8.0)
                p2 = predict_layer_param(b2, wl_192_640_t4["l2_meta"], 80.0, 90.0, ref_int, 14.0, 3.0, 4.0, 8.0)
                off_sweep.append({
                    "offset": off,
                    "meas_tot_mean": round(float(np.mean(tots)), 2),
                    "meas_tot_sd": round(float(np.std(tots, ddof=1)), 2),
                    "meas_l1_stall": round(float(np.mean(l1_st)), 2),
                    "meas_l2_stall": round(float(np.mean(l2_st)), 2),
                    "pred_tot": round(p1 + p2, 2),
                })

            pred_arr = np.array([r["pred_tot"] for r in off_sweep])
            meas_arr = np.array([r["meas_tot_mean"] for r in off_sweep])
            # Linear regression: meas = slope * pred + intercept
            slope, intercept = np.polyfit(pred_arr, meas_arr, 1)
            resid = meas_arr - (slope * pred_arr + intercept)
            s_err = np.sqrt(np.sum(resid ** 2) / (len(pred_arr) - 2))
            se_slope = s_err / np.sqrt(np.sum((pred_arr - np.mean(pred_arr)) ** 2))
            ci95_lo = slope - 2.042 * se_slope  # t_{0.975, df=30} = 2.042
            ci95_hi = slope + 2.042 * se_slope
            mae_cyc = float(np.mean(np.abs(meas_arr - pred_arr)))
            results["b5_4_offset32_90m"] = {
                "eng_meas_mhz": eng_m,
                "sdram_meas_mhz": sd_m,
                "slope_meas_vs_pred": round(float(slope), 4),
                "slope_ci95": [round(float(ci95_lo), 4), round(float(ci95_hi), 4)],
                "r_squared": round(float(np.corrcoef(pred_arr, meas_arr)[0, 1] ** 2), 4),
                "mae_cyc": round(mae_cyc, 2),
                "mean_sd_cyc": round(float(np.mean([r["meas_tot_sd"] for r in off_sweep])), 2),
                "records": off_sweep,
            }
            print(f"[B5.4] 32-offset slope={slope:.4f} (95% CI [{ci95_lo:.4f}, {ci95_hi:.4f}]), R2={results['b5_4_offset32_90m']['r_squared']:.4f}, MAE={mae_cyc:.2f} cyc")

            # B6.2/B6.3: Silicon DSE packing comparison (page-aligned addr0%32==0 vs unaligned addr0%32==1)
            dse_records = []
            for w in t4_wls:
                x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
                hpost_base = len(x_words)
                for i, xw in enumerate(x_words):
                    dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
                dev.flush()
                write_lut_256(dev, w["lut"], lut_bank=0)

                pack_res = {}
                for mode, off in (("aligned", 0), ("unaligned", 1)):
                    b1 = 32 + off
                    b2 = ((b1 + len(w["l1_words"]) + 31) // 32) * 32 + off
                    dev.sdram_write_bulk_words(b1, w["l1_words"])
                    dev.sdram_write_bulk_words(b2, w["l2_words"])
                    tots = []
                    for _ in range(N_REPEATS):
                        c1 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            4, w["d_model"], w["d_ff"], w["l1_meta"], b1, RD_LAT, 1, 0, hpost_base, 1,
                        )
                        c2 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            4, w["d_ff"], w["d_model"], w["l2_meta"], b2, RD_LAT, 0, hpost_base, 0, 0,
                        )
                        tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                    p1 = predict_layer_param(b1, w["l1_meta"], 80.0, 90.0, ref_int, 14.0, 3.0, 4.0, 8.0)
                    p2 = predict_layer_param(b2, w["l2_meta"], 80.0, 90.0, ref_int, 14.0, 3.0, 4.0, 8.0)
                    pack_res[mode] = {
                        "meas_mean": round(float(np.mean(tots)), 2),
                        "meas_sd": round(float(np.std(tots, ddof=1)), 2),
                        "pred": round(p1 + p2, 2),
                    }
                dse_records.append({
                    "shape": w["name"],
                    "aligned": pack_res["aligned"],
                    "unaligned": pack_res["unaligned"],
                    "pred_savings_cyc": round(pack_res["unaligned"]["pred"] - pack_res["aligned"]["pred"], 2),
                    "meas_savings_cyc": round(pack_res["unaligned"]["meas_mean"] - pack_res["aligned"]["meas_mean"], 2),
                })
                print(f"  [B6.2 DSE {w['name']}] pred savings={dse_records[-1]['pred_savings_cyc']} cyc, meas savings={dse_records[-1]['meas_savings_cyc']} cyc")
            results["b6_2_dse_packing_90m"] = dse_records
        finally:
            dev.close()

    finally:
        if SVF_DEFAULT.exists():
            flash_svf(SVF_DEFAULT)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[Done] Saved {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
