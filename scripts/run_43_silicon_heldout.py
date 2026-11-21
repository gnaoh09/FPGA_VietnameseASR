#!/usr/bin/env python3
"""Section 4.3 Extra Silicon Validation on EP4CE10E22C8N:
  1. Extrapolated held-out shapes (smaller 128->256->128 and K_chunks=3 320->960->320)
     on T_tile=4 at (80/100 MHz) and (70/100 MHz), n=10 repeats/pt, 0-LSB verified,
     with locked RTL predictions computed BEFORE measurement.
  2. T_tile=8 held-out with new engine clock (ENG=70 MHz, SDRAM=100 MHz, top_t8_e70_s100.svf)
     across 5 shapes (128->256->128, 192->384->192, 256->576->256, 256->960->256, 512->1536->512),
     n=10 repeats/pt, 0-LSB verified.

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
from gen_golden_vectors import apply_lut_256, dyadic_requantize_round_half_up, pack_words_128bit_i8
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks, run_bist_stats
from sdram_helpers import pack_layer_w4_sdram_words, run_sdram_stream_layer, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = BUILD_DIR / "output_files"
SVF_DEFAULT = OF_DIR / "top.svf"
OUT_JSON = BUILD_DIR / "validation_43_heldout_results.json"


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def make_ffn(d_model: int, d_ff: int, t_tile: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x_q = rng.integers(-64, 65, size=(t_tile, d_model), dtype=np.int8)
    w1_q = rng.integers(-7, 8, size=(d_ff, d_model), dtype=np.int8)
    b1_q = rng.integers(-4000, 4001, size=(d_ff,), dtype=np.int32)
    m1 = rng.integers(14000, 32000, size=(d_ff,), dtype=np.int32)
    sh1 = np.full((d_ff,), 20, dtype=np.int32)
    w2_q = rng.integers(-7, 8, size=(d_model, d_ff), dtype=np.int8)
    b2_q = rng.integers(-4000, 4001, size=(d_model,), dtype=np.int32)
    m2 = rng.integers(14000, 32000, size=(d_model,), dtype=np.int32)
    sh2 = np.full((d_model,), 20, dtype=np.int32)
    lut = np.array([max(-128, min(127, int(round(0.8 * x if x >= 0 else 0.1 * x)))) for x in range(-128, 128)], dtype=np.int8)
    h_pre_q = dyadic_requantize_round_half_up(x_q.astype(np.int32) @ w1_q.astype(np.int32).T + b1_q[None, :], m1, sh1)
    h_post_q = apply_lut_256(h_pre_q, lut)
    y_q = dyadic_requantize_round_half_up(h_post_q.astype(np.int32) @ w2_q.astype(np.int32).T + b2_q[None, :], m2, sh2)
    l1_words, l1_meta = pack_layer_w4_sdram_words(w1_q, b1_q, m1, sh1)
    l2_words, l2_meta = pack_layer_w4_sdram_words(w2_q, b2_q, m2, sh2)
    return {
        "d_model": d_model, "d_ff": d_ff, "t_tile": t_tile,
        "x_q": x_q, "h_post_q": h_post_q, "y_q": y_q, "lut": lut,
        "l1_words": l1_words, "l1_meta": l1_meta, "l2_words": l2_words, "l2_meta": l2_meta,
    }


def predict_rtl_gen(base_w128: int, meta: dict, t_tile: int, eng_mhz: float, sdram_mhz: float, ref_int: int = 1400, t_rfc: int = 8) -> float:
    n_tiles = meta["n_tiles"]
    k_chunks = meta["k_chunks"]
    k_chunk = meta["k_chunk"]
    pkt_words = meta["pkt_words"]
    c_tiles, c_dmas = [], []
    addr = base_w128
    for _ in range(n_tiles):
        for kc in range(k_chunks):
            has_epi = (kc == k_chunks - 1)
            c_tile = t_tile * k_chunk + (13 if has_epi else 6)
            pages = ((addr + pkt_words - 1) // 32) - (addr // 32) + 1
            sd_cyc = (8 * pkt_words + 14 * pages + 1) * (1.0 + t_rfc / float(ref_int)) + 3.0
            c_dma = sd_cyc * (eng_mhz / sdram_mhz) + 4.0
            c_tiles.append(c_tile)
            c_dmas.append(c_dma)
            addr += pkt_words
    stall = c_dmas[0] + sum(max(0.0, c_dmas[i] - c_tiles[i - 1]) for i in range(1, len(c_dmas)))
    return float(sum(c_tiles)) + stall


def run_suite(svf_path: Path, t_tile: int, eng_nom: float, sd_nom: float, shapes: list[tuple[int, int]]) -> list[dict]:
    wls = []
    sdram_words: list[int] = []
    for idx, (dm, df) in enumerate(shapes):
        wl = make_ffn(dm, df, t_tile, seed=20261160 + idx + t_tile * 100)
        b1 = len(sdram_words); sdram_words.extend(wl["l1_words"])
        b2 = len(sdram_words); sdram_words.extend(wl["l2_words"])
        p1 = predict_rtl_gen(b1, wl["l1_meta"], t_tile, eng_nom, sd_nom)
        p2 = predict_rtl_gen(b2, wl["l2_meta"], t_tile, eng_nom, sd_nom)
        wls.append({**wl, "l1_base": b1, "l2_base": b2, "pred_locked": round(p1 + p2, 2)})

    flash_svf(svf_path)
    dev = UsbBlasterJtag()
    rows = []
    try:
        eng_m, sd_m, _, _ = measure_pll_clocks(dev)
        bist = run_bist_stats(dev, 65536, sd_m)
        assert bist["err_cnt"] == 0
        dev.sdram_write_bulk_words(0, sdram_words)
        print(f"[{svf_path.name}] eng={eng_m:.3f}M, sdram={sd_m:.3f}M, BIST={bist['rd_MBps_1e6']} MB/s")
        for w in wls:
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
                    assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"0-LSB fail {w['d_model']}->{w['d_ff']}"
                tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                stalls.append(int(c1["stall_cycles"] + c2["stall_cycles"]))
            meas_m = float(np.mean(tots))
            meas_sd = float(np.std(tots, ddof=1))
            err_pct = abs(w["pred_locked"] - meas_m) / meas_m * 100.0
            row = {
                "svf": svf_path.name,
                "t_tile": t_tile,
                "eng_nom": eng_nom,
                "sd_nom": sd_nom,
                "eng_meas": eng_m,
                "sd_meas": sd_m,
                "shape": f"{w['d_model']}->{w['d_ff']}->{w['d_model']}",
                "l1_kchunks": w["l1_meta"]["k_chunks"],
                "l2_kchunks": w["l2_meta"]["k_chunks"],
                "pred_locked": w["pred_locked"],
                "meas_mean": round(meas_m, 2),
                "meas_sd": round(meas_sd, 2),
                "stall_pct": round(100.0 * float(np.mean(stalls)) / meas_m, 2),
                "err_pct": round(err_pct, 5),
            }
            rows.append(row)
            print(f"  [{row['shape']} kch=({row['l1_kchunks']},{row['l2_kchunks']})] meas={meas_m:.1f}+/-{meas_sd:.2f} | pred={w['pred_locked']} ({err_pct:.4f}%)")
    finally:
        dev.close()
    return rows


def main() -> int:
    t4_heldout_shapes = [(128, 256), (320, 960)]
    t8_heldout_shapes = [(128, 256), (192, 384), (256, 576), (256, 960), (512, 1536)]
    res = {}
    try:
        res["t4_e80_s100_heldout"] = run_suite(SVF_DEFAULT, 4, 80.0, 100.0, t4_heldout_shapes)
        res["t4_e70_s100_heldout"] = run_suite(OF_DIR / "top_eng70_sd100.svf", 4, 70.0, 100.0, t4_heldout_shapes)
        res["t8_e70_s100_heldout"] = run_suite(OF_DIR / "top_t8_e70_s100.svf", 8, 70.0, 100.0, t8_heldout_shapes)
        OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved: {OUT_JSON}")
    finally:
        if SVF_DEFAULT.exists():
            flash_svf(SVF_DEFAULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
