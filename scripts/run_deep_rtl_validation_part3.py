#!/usr/bin/env python3
"""Deep RTL & Hardware Validation Part 3 on EP4CE10E22C8N:
  1. T_tile = 8 frequency sweep across {40, 50, 60, 80, 100} MHz SDRAM (@ 80 MHz Engine)
     for 4 shapes (192->384->192, 256->576->256, 256->960->256, 512->1536->512), n=10 repeats/pt,
     0-LSB verification, 1 MiB BIST, and RTL predictor comparison (C_tile = 8*K_chunk + epi).
  2. Full 0..31 offset sweep with n=10 repeats/offset on 192->640->192 @ 90 MHz:
     computes linear regression slope (meas vs pred), MAE (cycles), and MAE/SD ratio.
  3. Controller/Layer generalization on 80/100 MHz (top.svf):
     - RD_LAT = 4 vs RD_LAT = 6 (n=10 each) on 192->640->192
     - Attention projection linear layer (256->256, lut_mode=0, T=4, n=10, 0-LSB).

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
from gen_golden_vectors import dyadic_requantize_round_half_up, pack_words_128bit_i8
from run_deep_rtl_validation_part1 import run_ffn_pair, setup_shape
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks, run_bist_stats
from run_oos_stall_validation import make_synthetic_integer_ffn, predict_layer_rtl
from run_t_tile8_ep4ce10 import SHAPES as T8_SHAPES, make_ffn_t8, predict_layer_rtl_t8
from sdram_helpers import pack_layer_w4_sdram_words, run_sdram_stream_layer, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = BUILD_DIR / "output_files"
SVF_DEFAULT = OF_DIR / "top.svf"
OUT_JSON = BUILD_DIR / "deep_rtl_validation_part3.json"

T8_BITSTREAMS = [
    {"mhz": 40,  "svf": OF_DIR / "top_t8_sd40.svf"},
    {"mhz": 50,  "svf": OF_DIR / "top_t8_sd50.svf"},
    {"mhz": 60,  "svf": OF_DIR / "top_t8_sd60.svf"},
    {"mhz": 80,  "svf": OF_DIR / "top_t8_sd80.svf"},
    {"mhz": 100, "svf": OF_DIR / "top_t8.svf"},
]


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_data: dict = {"device": "EP4CE10E22C8N"}

    # Prepare T_tile=8 workloads
    t8_wls = []
    t8_sdram_words: list[int] = []
    for idx, spec in enumerate(T8_SHAPES):
        wl = make_ffn_t8(spec["d_model"], spec["d_ff"], seed=20261140 + idx)
        b1 = len(t8_sdram_words)
        t8_sdram_words.extend(wl["l1_words"])
        b2 = len(t8_sdram_words)
        t8_sdram_words.extend(wl["l2_words"])
        t8_wls.append({**spec, **wl, "l1_base": b1, "l2_base": b2})

    try:
        # 1) T_tile=8 frequency sweep across {40, 50, 60, 80, 100} MHz
        t8_sweep = []
        for bs in T8_BITSTREAMS:
            mhz = bs["mhz"]
            flash_svf(bs["svf"])
            dev = UsbBlasterJtag()
            try:
                eng_m, sd_m, _, _ = measure_pll_clocks(dev)
                bist = run_bist_stats(dev, 65536, sd_m)
                assert bist["err_cnt"] == 0
                dev.sdram_write_bulk_words(0, t8_sdram_words)
                ref_int = mhz * 14
                print(f"[T8 @ {mhz}M] eng={eng_m:.3f}M, sdram={sd_m:.3f}M, BIST={bist['rd_MBps_1e6']} MB/s")

                for w in t8_wls:
                    x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
                    hpost_base = len(x_words)
                    for i, xw in enumerate(x_words):
                        dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
                    dev.flush()
                    write_lut_256(dev, w["lut"], lut_bank=0)

                    tots, stalls = [], []
                    for rep in range(10):
                        c1 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            8, w["d_model"], w["d_ff"], w["l1_meta"], w["l1_base"],
                            4, 1, 0, hpost_base, 1
                        )
                        c2 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            8, w["d_ff"], w["d_model"], w["l2_meta"], w["l2_base"],
                            4, 0, hpost_base, 0, 0
                        )
                        if rep == 0:
                            y_exp = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]
                            assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"T8 0-LSB fail {mhz} {w['name']}"
                        tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                        stalls.append(int(c1["stall_cycles"] + c2["stall_cycles"]))

                    p1 = predict_layer_rtl_t8(w["l1_base"], w["l1_meta"], float(mhz), ref_int, 8, 80.0)
                    p2 = predict_layer_rtl_t8(w["l2_base"], w["l2_meta"], float(mhz), ref_int, 8, 80.0)
                    pred_tot = p1["pred_total_cyc"] + p2["pred_total_cyc"]
                    pred_st = p1["pred_stall_cyc"] + p2["pred_stall_cyc"]
                    meas_m = float(np.mean(tots))
                    meas_sd = float(np.std(tots, ddof=1))
                    err_pct = abs(pred_tot - meas_m) / meas_m * 100.0
                    t8_sweep.append({
                        "mhz": mhz,
                        "eng_meas": eng_m,
                        "sd_meas": sd_m,
                        "bist_rd_MBps": bist["rd_MBps_1e6"],
                        "shape": w["name"],
                        "meas_tot_mean": round(meas_m, 2),
                        "meas_tot_sd": round(meas_sd, 2),
                        "meas_stall_mean": round(float(np.mean(stalls)), 2),
                        "meas_stall_pct": round(100.0 * float(np.mean(stalls)) / meas_m, 2),
                        "pred_tot": pred_tot,
                        "pred_stall": pred_st,
                        "err_pct": round(err_pct, 5),
                    })
                    print(f"  [{w['name']}] meas={meas_m:.1f}+/-{meas_sd:.2f} | pred={pred_tot} ({err_pct:.4f}%) | stall={100*np.mean(stalls)/meas_m:.2f}%")
            finally:
                dev.close()
        out_data["t8_freq_sweep"] = t8_sweep

        # 2) Full 0..31 offset sweep with n=10 at 90 MHz (192->640->192)
        wl_640 = make_synthetic_integer_ffn(192, 640, seed=20261121)
        flash_svf(OF_DIR / "top_sdram90.svf")
        dev = UsbBlasterJtag()
        try:
            eng90, sd90, _, _ = measure_pll_clocks(dev)
            setup_shape(dev, wl_640)
            dev.sdram_write_bulk_words(16384, wl_640["l1_words"])
            dev.sdram_write_bulk_words(32768, wl_640["l2_words"])
            sweep_n10 = []
            for off in range(32):
                b1, b2 = 16384 + off, 32768 + off
                runs = [run_ffn_pair(dev, wl_640, b1, b2) for _ in range(10)]
                tots = [r["total_cycles"] for r in runs]
                p1 = predict_layer_rtl(b1, wl_640["l1_meta"], 90.0, 1260, 8, 80.0)
                p2 = predict_layer_rtl(b2, wl_640["l2_meta"], 90.0, 1260, 8, 80.0)
                sweep_n10.append({
                    "offset": off,
                    "pred_tot": p1["pred_total_cyc"] + p2["pred_total_cyc"],
                    "meas_tot_mean": round(float(np.mean(tots)), 2),
                    "meas_tot_sd": round(float(np.std(tots, ddof=1)), 2),
                })
            preds = np.array([x["pred_tot"] for x in sweep_n10], dtype=np.float64)
            meass = np.array([x["meas_tot_mean"] for x in sweep_n10], dtype=np.float64)
            sds = np.array([x["meas_tot_sd"] for x in sweep_n10], dtype=np.float64)
            slope, intercept = np.polyfit(preds, meass, 1)
            mae_cyc = float(np.mean(np.abs(meass - preds)))
            out_data["offset_0_31_n10_90mhz"] = {
                "eng_meas": eng90,
                "sd_meas": sd90,
                "slope_meas_vs_pred": round(float(slope), 4),
                "intercept_cyc": round(float(intercept), 2),
                "mae_cyc": round(mae_cyc, 2),
                "mean_sd_cyc": round(float(np.mean(sds)), 2),
                "mae_over_sd": round(mae_cyc / float(np.mean(sds)), 2),
                "rows": sweep_n10,
            }
            print(f"[Offset 0..31 n=10 @ 90M] slope={slope:.4f}, MAE={mae_cyc:.2f} cyc, mean_SD={np.mean(sds):.2f} cyc")
        finally:
            dev.close()

        # 3) Generalization: RD_LAT=4 vs RD_LAT=6 + Attention projection (256->256, lut_mode=0) at 80/100 MHz
        flash_svf(SVF_DEFAULT)
        dev = UsbBlasterJtag()
        try:
            eng100, sd100, _, _ = measure_pll_clocks(dev)
            setup_shape(dev, wl_640)
            dev.sdram_write_bulk_words(16384, wl_640["l1_words"])
            dev.sdram_write_bulk_words(32768, wl_640["l2_words"])
            rdlat_res = {}
            for rlat in (4, 6):
                tots = []
                for _ in range(10):
                    c1 = run_sdram_stream_layer(
                        dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                        4, 192, 640, wl_640["l1_meta"], 16384, rlat, 1, 0, 48, 1
                    )
                    c2 = run_sdram_stream_layer(
                        dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                        4, 640, 192, wl_640["l2_meta"], 32768, rlat, 0, 48, 0, 0
                    )
                    tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                rdlat_res[f"rd_lat_{rlat}"] = {
                    "mean": round(float(np.mean(tots)), 2),
                    "sd": round(float(np.std(tots, ddof=1)), 2),
                }

            # Attention projection 256->256 (lut_mode=0, T=4)
            rng = np.random.default_rng(20261155)
            x_a = rng.integers(-64, 65, size=(4, 256), dtype=np.int8)
            w_a = rng.integers(-7, 8, size=(256, 256), dtype=np.int8)
            b_a = rng.integers(-4000, 4001, size=(256,), dtype=np.int32)
            m_a = rng.integers(14000, 32000, size=(256,), dtype=np.int32)
            sh_a = np.full((256,), 20, dtype=np.int32)
            y_a = dyadic_requantize_round_half_up(x_a.astype(np.int32) @ w_a.astype(np.int32).T + b_a[None, :], m_a, sh_a)
            wa_words, wa_meta = pack_layer_w4_sdram_words(w_a, b_a, m_a, sh_a)
            dev.sdram_write_bulk_words(0, wa_words)
            for i, xw in enumerate([int(s, 16) for s in pack_words_128bit_i8(x_a)]):
                dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
            dev.flush()
            attn_tots = []
            for rep in range(10):
                ca = run_sdram_stream_layer(
                    dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                    4, 256, 256, wa_meta, 0, 4, 0, 0, 0, 0
                )
                if rep == 0:
                    ya_exp = [int(s, 16) for s in pack_words_128bit_i8(y_a)]
                    assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(ya_exp)), "Attn 256->256 0-LSB fail"
                attn_tots.append(int(ca["total_cycles"]))
            pred_attn = predict_layer_rtl(0, wa_meta, 100.0, 1400, 8, 80.0)["pred_total_cyc"]
            out_data["generalization_100mhz"] = {
                "eng_meas": eng100,
                "sd_meas": sd100,
                "rd_lat_ablation": rdlat_res,
                "attn_proj_256_256": {
                    "meas_mean": round(float(np.mean(attn_tots)), 2),
                    "meas_sd": round(float(np.std(attn_tots, ddof=1)), 2),
                    "pred_rtl": pred_attn,
                    "err_pct": round(abs(pred_attn - float(np.mean(attn_tots))) / float(np.mean(attn_tots)) * 100.0, 5),
                },
            }
            print("[Generalization 100M]", json.dumps(out_data["generalization_100mhz"]))
        finally:
            dev.close()

        OUT_JSON.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"Saved: {OUT_JSON}")
    finally:
        if SVF_DEFAULT.exists():
            flash_svf(SVF_DEFAULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
