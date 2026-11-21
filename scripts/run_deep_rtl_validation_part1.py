#!/usr/bin/env python3
"""Deep RTL & Hardware Validation Part 1 on EP4CE10E22C8N:
  1. Refresh-phase experiment (N=100 pairs = 200 runs of 192->640->192 at 80/90 MHz,
     logging sdram_cyc_cnt % 1260 before L1 and L2).
  2. Full 32-offset sweep (0..31, n=3 each) on 192->640->192 at 90 MHz and 100 MHz,
     plus 192->512->192 at 90 MHz.
  3. Bank/Row diversity test (Bank 0/1/2/3 + Row+1/Row+100, n=10 each) at 90 MHz.

Always restores build/output_files/top.svf in finally:.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpga_jtag_runner import (
    OP_COMMIT_ACT,
    OP_START,
    OP_WRITE_REG,
    UsbBlasterJtag,
    pack_cfg1,
    pack_cfg2,
)
from gen_golden_vectors import pack_words_128bit_i8
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks, run_bist_stats
from run_oos_stall_validation import make_synthetic_integer_ffn, predict_layer_rtl
from sdram_helpers import pack_cfg4, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = BUILD_DIR / "output_files"
SVF_90 = OF_DIR / "top_sdram90.svf"
SVF_100 = OF_DIR / "top.svf"
OUT_JSON = BUILD_DIR / "deep_rtl_validation_part1.json"

T_TILE = 4
RD_LAT = 4


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def run_layer_with_phase(
    dev: UsbBlasterJtag,
    dim_t: int,
    k_dim: int,
    n_dim: int,
    layer_meta: dict,
    sdram_base_w128: int,
    lut_mode: int,
    act_base: int,
    dst_base: int,
    dst_is_act: int,
) -> dict:
    n_tiles = layer_meta["n_tiles"]
    k_chunks = layer_meta["k_chunks"]
    k_chunk = layer_meta["k_chunk"]
    w_words = layer_meta["w_words_per_chunk"]

    dev.queue_cmd(OP_WRITE_REG, 0, pack_cfg0_ext(0, 1, lut_mode, dim_t, k_chunk, 16, acc_mode=0))
    dev.queue_cmd(OP_WRITE_REG, 1, pack_cfg1(16, act_base & 0xFF, 0, dst_base & 0xFF))
    dev.queue_cmd(OP_WRITE_REG, 2, pack_cfg2(0, 0, 0, dst_base & 0xFF))
    dev.queue_cmd(
        OP_WRITE_REG,
        3,
        pack_cfg3_ext(
            0, k_chunk, act_base, dst_base,
            dst_stride=n_dim // 16, stress_loop=0,
            act_stride=k_dim // 16, dst_is_act=dst_is_act,
        ),
    )
    dev.queue_cmd(OP_WRITE_REG, 4, pack_cfg4(1, n_tiles, k_chunks, RD_LAT, w_words))
    dev.queue_cmd(OP_WRITE_REG, 5, sdram_base_w128 & 0x7FFFF)
    dev.flush()

    sd_cyc_pre = dev.read_resp(0x200A)
    dev.queue_cmd(OP_START, 0, 0)
    dev.flush()
    cnt = dev.wait_done(timeout_s=5.0)
    r9 = dev.read_resp(0x2009)
    r11 = dev.read_resp(0x200B)
    return {
        "sd_cyc_pre": int(sd_cyc_pre),
        "total_cycles": int(cnt["total_cycles"]),
        "mac_cycles": int(cnt["mac_cycles"]),
        "stall_cycles": int((r9 >> 32) & 0xFFFFFFFF),
        "dma_cycles": int((r11 >> 32) & 0xFFFFFFFF),
    }


def setup_shape(dev: UsbBlasterJtag, wl: dict) -> None:
    x_words = [int(s, 16) for s in pack_words_128bit_i8(wl["x_q"])]
    for i, xw in enumerate(x_words):
        dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
    dev.flush()
    write_lut_256(dev, wl["lut"], lut_bank=0)


def run_ffn_pair(dev: UsbBlasterJtag, wl: dict, l1_base: int, l2_base: int) -> dict:
    hpost_base = (wl["d_model"] // 16) * T_TILE
    c1 = run_layer_with_phase(
        dev, T_TILE, wl["d_model"], wl["d_ff"], wl["l1_meta"], l1_base, 1, 0, hpost_base, 1
    )
    c2 = run_layer_with_phase(
        dev, T_TILE, wl["d_ff"], wl["d_model"], wl["l2_meta"], l2_base, 0, hpost_base, 0, 0
    )
    return {
        "l1": c1,
        "l2": c2,
        "total_cycles": c1["total_cycles"] + c2["total_cycles"],
        "stall_cycles": c1["stall_cycles"] + c2["stall_cycles"],
    }


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    wl_640 = make_synthetic_integer_ffn(192, 640, seed=20261121)
    wl_512 = make_synthetic_integer_ffn(192, 512, seed=20261122)

    results: dict = {"device": "EP4CE10E22C8N"}

    try:
        # ---------------- 90 MHz SDRAM ----------------
        flash_svf(SVF_90)
        dev = UsbBlasterJtag()
        try:
            eng90, sd90, _, _ = measure_pll_clocks(dev)
            bist90 = run_bist_stats(dev, 65536, sd90)
            assert bist90["err_cnt"] == 0
            print(f"[90M] eng={eng90:.3f} MHz, sdram={sd90:.3f} MHz, BIST={bist90['rd_MBps_1e6']} MB/s")

            # Upload 192->640->192 at base=16384+1 and 32768+1 for 0-LSB sanity check
            setup_shape(dev, wl_640)
            dev.sdram_write_bulk_words(16385, wl_640["l1_words"])
            dev.sdram_write_bulk_words(32769, wl_640["l2_words"])
            r_chk = run_ffn_pair(dev, wl_640, 16385, 32769)
            y_exp = [int(s, 16) for s in pack_words_128bit_i8(wl_640["y_q"])]
            assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), "0-LSB check failed at 90M!"
            print(f"[90M 0-LSB PASS] 192->640->192 @ off=1: tot={r_chk['total_cycles']}")

            # 1) Refresh-phase experiment: N=100 pairs (200 runs)
            print("[90M] Running 100 pairs (200 runs) for refresh-phase analysis...")
            phase_runs = []
            for p_idx in range(100):
                order = [(1, 16385, 32769), (15, 16399, 32783)] if (p_idx % 2 == 0) else [(15, 16399, 32783), (1, 16385, 32769)]
                for off, b1, b2 in order:
                    res = run_ffn_pair(dev, wl_640, b1, b2)
                    phase_runs.append({
                        "pair": p_idx,
                        "offset": off,
                        "l1_phase": res["l1"]["sd_cyc_pre"] % 1260,
                        "l2_phase": res["l2"]["sd_cyc_pre"] % 1260,
                        "l1_tot": res["l1"]["total_cycles"],
                        "l1_st": res["l1"]["stall_cycles"],
                        "l2_tot": res["l2"]["total_cycles"],
                        "l2_st": res["l2"]["stall_cycles"],
                        "tot": res["total_cycles"],
                    })
            o1 = [r for r in phase_runs if r["offset"] == 1]
            o15 = [r for r in phase_runs if r["offset"] == 15]
            d_tot = [a["tot"] - b["tot"] for a, b in zip(o1, o15)]
            d_l1 = [a["l1_tot"] - b["l1_tot"] for a, b in zip(o1, o15)]
            d_l2 = [a["l2_tot"] - b["l2_tot"] for a, b in zip(o1, o15)]
            print(
                f"[90M Phase N=100 pairs] d_tot={np.mean(d_tot):.2f}+/-{np.std(d_tot, ddof=1):.2f} "
                f"| d_l1={np.mean(d_l1):.2f}+/-{np.std(d_l1, ddof=1):.2f} "
                f"| d_l2={np.mean(d_l2):.2f}+/-{np.std(d_l2, ddof=1):.2f}"
            )

            # 2) Full 0..31 offset sweep on 192->640->192 (n=3 per offset)
            print("[90M] Full 0..31 offset sweep on 192->640->192 (n=3)...")
            sweep_640_90 = []
            for off in range(32):
                b1, b2 = 16384 + off, 32768 + off
                runs = [run_ffn_pair(dev, wl_640, b1, b2) for _ in range(3)]
                p1 = predict_layer_rtl(b1, wl_640["l1_meta"], 90.0, 1260, 8, 80.0)
                p2 = predict_layer_rtl(b2, wl_640["l2_meta"], 90.0, 1260, 8, 80.0)
                sweep_640_90.append({
                    "offset": off,
                    "pred_tot": p1["pred_total_cyc"] + p2["pred_total_cyc"],
                    "pred_l1": p1["pred_total_cyc"],
                    "pred_l2": p2["pred_total_cyc"],
                    "meas_tot_mean": round(float(np.mean([r["total_cycles"] for r in runs])), 2),
                    "meas_tot_sd": round(float(np.std([r["total_cycles"] for r in runs], ddof=1)), 2),
                    "meas_l1_mean": round(float(np.mean([r["l1"]["total_cycles"] for r in runs])), 2),
                    "meas_l2_mean": round(float(np.mean([r["l2"]["total_cycles"] for r in runs])), 2),
                })

            # 3) Full 0..31 offset sweep on 192->512->192 (n=3 per offset)
            print("[90M] Full 0..31 offset sweep on 192->512->192 (n=3)...")
            setup_shape(dev, wl_512)
            sweep_512_90 = []
            for off in range(32):
                b1, b2 = 16384 + off, 32768 + off
                runs = [run_ffn_pair(dev, wl_512, b1, b2) for _ in range(3)]
                p1 = predict_layer_rtl(b1, wl_512["l1_meta"], 90.0, 1260, 8, 80.0)
                p2 = predict_layer_rtl(b2, wl_512["l2_meta"], 90.0, 1260, 8, 80.0)
                sweep_512_90.append({
                    "offset": off,
                    "pred_tot": p1["pred_total_cyc"] + p2["pred_total_cyc"],
                    "meas_tot_mean": round(float(np.mean([r["total_cycles"] for r in runs])), 2),
                    "meas_tot_sd": round(float(np.std([r["total_cycles"] for r in runs], ddof=1)), 2),
                    "meas_l1_mean": round(float(np.mean([r["l1"]["total_cycles"] for r in runs])), 2),
                    "meas_l2_mean": round(float(np.mean([r["l2"]["total_cycles"] for r in runs])), 2),
                })

            # 4) Bank/Row diversity (n=10 each) on 192->640->192 at offset 1 vs 15
            print("[90M] Bank/Row diversity test (n=10 each)...")
            setup_shape(dev, wl_640)
            bank_row_configs = [
                ("bank0_row0", 16384, 32768),
                ("bank1_row0", 16384 + 32, 32768 + 32),
                ("bank2_row0", 16384 + 64, 32768 + 64),
                ("bank3_row0", 16384 + 96, 32768 + 96),
                ("bank0_row1", 16384 + 128, 32768 + 128),
                ("bank0_row100", 16384 + 12800, 32768 + 12800),
            ]
            bank_row_res = []
            for label, base1, base2 in bank_row_configs:
                d_list, l1_list, l2_list = [], [], []
                for _ in range(10):
                    r1 = run_ffn_pair(dev, wl_640, base1 + 1, base2 + 1)
                    r15 = run_ffn_pair(dev, wl_640, base1 + 15, base2 + 15)
                    d_list.append(r1["total_cycles"] - r15["total_cycles"])
                    l1_list.append(r1["l1"]["total_cycles"] - r15["l1"]["total_cycles"])
                    l2_list.append(r1["l2"]["total_cycles"] - r15["l2"]["total_cycles"])
                bank_row_res.append({
                    "label": label,
                    "d_tot_mean": round(float(np.mean(d_list)), 2),
                    "d_tot_sd": round(float(np.std(d_list, ddof=1)), 2),
                    "d_l1_mean": round(float(np.mean(l1_list)), 2),
                    "d_l2_mean": round(float(np.mean(l2_list)), 2),
                })
                print(f"  [{label}] d_tot={np.mean(d_list):+.2f}+/-{np.std(d_list, ddof=1):.2f} (L1={np.mean(l1_list):+.2f}, L2={np.mean(l2_list):+.2f})")

            results["sdram_90mhz"] = {
                "eng_mhz": eng90,
                "sdram_mhz": sd90,
                "bist_rd_MBps": bist90["rd_MBps_1e6"],
                "phase_100pairs": {
                    "d_tot_mean": round(float(np.mean(d_tot)), 2),
                    "d_tot_sd": round(float(np.std(d_tot, ddof=1)), 2),
                    "d_l1_mean": round(float(np.mean(d_l1)), 2),
                    "d_l1_sd": round(float(np.std(d_l1, ddof=1)), 2),
                    "d_l2_mean": round(float(np.mean(d_l2)), 2),
                    "d_l2_sd": round(float(np.std(d_l2, ddof=1)), 2),
                    "runs": phase_runs,
                },
                "sweep_640_0_31": sweep_640_90,
                "sweep_512_0_31": sweep_512_90,
                "bank_row_diversity": bank_row_res,
            }
        finally:
            dev.close()

        # ---------------- 100 MHz SDRAM ----------------
        flash_svf(SVF_100)
        dev = UsbBlasterJtag()
        try:
            eng100, sd100, _, _ = measure_pll_clocks(dev)
            print(f"[100M] eng={eng100:.3f} MHz, sdram={sd100:.3f} MHz")
            setup_shape(dev, wl_640)
            sweep_640_100 = []
            for off in range(32):
                b1, b2 = 16384 + off, 32768 + off
                runs = [run_ffn_pair(dev, wl_640, b1, b2) for _ in range(3)]
                p1 = predict_layer_rtl(b1, wl_640["l1_meta"], 100.0, 1400, 8, 80.0)
                p2 = predict_layer_rtl(b2, wl_640["l2_meta"], 100.0, 1400, 8, 80.0)
                sweep_640_100.append({
                    "offset": off,
                    "pred_tot": p1["pred_total_cyc"] + p2["pred_total_cyc"],
                    "meas_tot_mean": round(float(np.mean([r["total_cycles"] for r in runs])), 2),
                    "meas_tot_sd": round(float(np.std([r["total_cycles"] for r in runs], ddof=1)), 2),
                })
            results["sdram_100mhz"] = {
                "eng_mhz": eng100,
                "sdram_mhz": sd100,
                "sweep_640_0_31": sweep_640_100,
            }
        finally:
            dev.close()

        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved: {OUT_JSON}")
    finally:
        if SVF_100.exists():
            flash_svf(SVF_100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
