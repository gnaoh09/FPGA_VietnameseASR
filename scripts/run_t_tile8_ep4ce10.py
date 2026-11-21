#!/usr/bin/env python3
"""Hardware evaluation of MAX_T_TILE=8 (top_t8.svf) on Cyclone IV EP4CE10E22C8N
at 80 MHz Engine / 100 MHz SDRAM.

Runs on live silicon:
  1. JTAG link check, on-chip PLL frequency measurement, 1 MiB OP_BIST (0 errors).
  2. Real-size Zipformer FFN (512 -> 1536 -> 512, W4A8, T_tile=8) via SDRAM Ping-Pong DMA
     with 0 LSB mismatch verification on both h_post (768 words = 12,288 B) and y_out (256 words = 4,096 B).
  3. Three Zipformer-30M FFN shapes (192->384->192, 256->576->256, 256->960->256) at T_tile=8
     with 0 LSB mismatch verification and comparison against the RTL schedule predictor
     (c_tile = 8 * k_chunk + (13 if has_epi else 6)) and T_tile=4 baseline.

Always restores build/output_files/top.svf (T_tile=4) in finally:.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpga_jtag_runner import (
    OP_COMMIT_ACT,
    OP_WRITE_REG,
    UsbBlasterJtag,
    pack_cfg1,
    pack_cfg2,
)
from gen_golden_vectors import (
    apply_lut_256,
    dyadic_requantize_round_half_up,
    pack_words_128bit_i8,
)
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks, run_bist_stats
from sdram_helpers import (
    pack_layer_w4_sdram_words,
    run_sdram_stream_layer,
    write_lut_256,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = BUILD_DIR / "output_files"
TOP_T8_SVF = OF_DIR / "top_t8.svf"
TOP_T4_SVF = OF_DIR / "top.svf"
OUT_JSON = BUILD_DIR / "t_tile8_ep4ce10_results.json"

T_TILE = 8
RD_LAT = 4
REF_INTERVAL_100M = 1400
T_RFC_100M = 8

SHAPES = [
    {"name": "ffn_192_384_192", "label": "192->384->192 (stage_0)", "d_model": 192, "d_ff": 384},
    {"name": "ffn_256_576_256", "label": "256->576->256 (stage_1..5 FFN1)", "d_model": 256, "d_ff": 576},
    {"name": "ffn_256_960_256", "label": "256->960->256 (stage_1..5 FFN3)", "d_model": 256, "d_ff": 960},
    {"name": "ffn_512_1536_512", "label": "512->1536->512 (full-size FFN)", "d_model": 512, "d_ff": 1536},
]


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def make_ffn_t8(d_model: int, d_ff: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x_q = rng.integers(-64, 65, size=(T_TILE, d_model), dtype=np.int8)
    w1_q = rng.integers(-7, 8, size=(d_ff, d_model), dtype=np.int8)
    b1_q = rng.integers(-4000, 4001, size=(d_ff,), dtype=np.int32)
    m1 = rng.integers(14000, 32000, size=(d_ff,), dtype=np.int32)
    sh1 = np.full((d_ff,), 20, dtype=np.int32)

    w2_q = rng.integers(-7, 8, size=(d_model, d_ff), dtype=np.int8)
    b2_q = rng.integers(-4000, 4001, size=(d_model,), dtype=np.int32)
    m2 = rng.integers(14000, 32000, size=(d_model,), dtype=np.int32)
    sh2 = np.full((d_model,), 20, dtype=np.int32)

    lut = np.array(
        [max(-128, min(127, int(round(0.8 * x if x >= 0 else 0.1 * x)))) for x in range(-128, 128)],
        dtype=np.int8,
    )

    acc1 = x_q.astype(np.int32) @ w1_q.astype(np.int32).T + b1_q[None, :]
    h_pre_q = dyadic_requantize_round_half_up(acc1, m1, sh1)
    h_post_q = apply_lut_256(h_pre_q, lut)
    acc2 = h_post_q.astype(np.int32) @ w2_q.astype(np.int32).T + b2_q[None, :]
    y_q = dyadic_requantize_round_half_up(acc2, m2, sh2)

    l1_words, l1_meta = pack_layer_w4_sdram_words(w1_q, b1_q, m1, sh1)
    l2_words, l2_meta = pack_layer_w4_sdram_words(w2_q, b2_q, m2, sh2)

    return {
        "d_model": d_model,
        "d_ff": d_ff,
        "x_q": x_q,
        "h_post_q": h_post_q,
        "y_q": y_q,
        "lut": lut,
        "l1_words": l1_words,
        "l1_meta": l1_meta,
        "l2_words": l2_words,
        "l2_meta": l2_meta,
    }


def predict_layer_rtl_t8(
    base_w128: int,
    layer_meta: dict,
    nom_sdram_mhz: float = 100.0,
    ref_interval: int = REF_INTERVAL_100M,
    t_rfc: int = T_RFC_100M,
    eng_mhz: float = 80.0,
) -> dict:
    n_tiles = int(layer_meta["n_tiles"])
    k_chunks = int(layer_meta["k_chunks"])
    k_chunk = int(layer_meta["k_chunk"])
    pkt_words = int(layer_meta["pkt_words"])
    num_chunks = n_tiles * k_chunks

    c_tiles: list[int] = []
    c_dmas: list[float] = []
    addr = base_w128
    for _nt in range(n_tiles):
        for kc in range(k_chunks):
            has_epi = (kc == k_chunks - 1)
            c_tile = T_TILE * k_chunk + (13 if has_epi else 6)
            pages = ((addr + pkt_words - 1) // 32) - (addr // 32) + 1
            sd_cyc = (8 * pkt_words + 14 * pages + 1) * (1.0 + t_rfc / float(ref_interval)) + 3.0
            c_dma = sd_cyc * (eng_mhz / nom_sdram_mhz) + 4.0
            c_tiles.append(c_tile)
            c_dmas.append(c_dma)
            addr += pkt_words

    stall = c_dmas[0]
    for idx in range(1, num_chunks):
        stall += max(0.0, c_dmas[idx] - c_tiles[idx - 1])
    active = float(sum(c_tiles))
    total = active + stall
    return {
        "pred_active_cyc": int(round(active)),
        "pred_stall_cyc": int(round(stall)),
        "pred_total_cyc": int(round(total)),
        "pred_stall_pct": round(100.0 * stall / total, 2),
    }


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    workloads = []
    all_sdram_words: list[int] = []
    for idx, spec in enumerate(SHAPES):
        wl = make_ffn_t8(spec["d_model"], spec["d_ff"], seed=20261120 + idx)
        l1_base = len(all_sdram_words)
        all_sdram_words.extend(wl["l1_words"])
        l2_base = len(all_sdram_words)
        all_sdram_words.extend(wl["l2_words"])
        workloads.append({**spec, **wl, "l1_base": l1_base, "l2_base": l2_base})

    print(f"Total SDRAM payload for all 4 T_tile=8 shapes: {len(all_sdram_words)} x 128b ({len(all_sdram_words)*16/1024:.1f} KiB)")

    try:
        flash_svf(TOP_T8_SVF)
        dev = UsbBlasterJtag()
        try:
            id_reg = dev.read_resp(0x2000)
            magic = (id_reg >> 32) & 0xFFFFFFFF
            num_lanes = (id_reg >> 16) & 0xFFFF
            ver = id_reg & 0xFFFF
            assert magic == 0x5A463445, f"Unexpected FPGA MAGIC 0x{magic:08x}"
            id_info = {"magic": magic, "num_lanes": num_lanes, "ver": ver}
            print(f"[Board] MAGIC=0x{magic:08x}, NUM_LANES={num_lanes}, VER=0x{ver:04x}")
            eng_mhz, sd_mhz, eng_samples, sd_samples = measure_pll_clocks(dev)
            clks = {"eng_mhz": eng_mhz, "sdram_mhz": sd_mhz, "eng_samples": eng_samples, "sdram_samples": sd_samples}
            print(f"[Clocks] eng={eng_mhz:.3f} MHz, sdram={sd_mhz:.3f} MHz")
            bist = run_bist_stats(dev, 65536, sd_mhz)
            print(f"[BIST 1 MiB] err={bist['err_cnt']}, wr={bist['wr_MBps_1e6']} MB/s, rd={bist['rd_MBps_1e6']} MB/s")
            assert bist["err_cnt"] == 0, f"BIST failed with {bist['err_cnt']} errors!"

            t_up = dev.sdram_write_bulk_words(0, all_sdram_words)
            print(f"[SDRAM Upload] {len(all_sdram_words)} words in {t_up:.2f} s")

            shape_results = []
            for w in workloads:
                d_model = w["d_model"]
                d_ff = w["d_ff"]
                x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
                hpost_base = len(x_words)
                hpost_exp_words = [int(s, 16) for s in pack_words_128bit_i8(w["h_post_q"])]
                y_exp_words = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]

                assert hpost_base + len(hpost_exp_words) <= 1024, "act_mem overflow!"
                assert len(y_exp_words) <= 256, "out_mem overflow!"

                for i, xw in enumerate(x_words):
                    dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
                dev.flush()
                write_lut_256(dev, w["lut"], lut_bank=0)

                c1 = run_sdram_stream_layer(
                    dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                    T_TILE, d_model, d_ff, w["l1_meta"], w["l1_base"], RD_LAT, 1, 0, hpost_base, 1
                )
                c2 = run_sdram_stream_layer(
                    dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                    T_TILE, d_ff, d_model, w["l2_meta"], w["l2_base"], RD_LAT, 0, hpost_base, 0, 0
                )

                # Verify 0 LSB mismatch on BOTH h_post (mem_id=1: act_mem) and y_out (mem_id=0: out_mem)
                hpost_mismatch = 0
                for i, exp in enumerate(hpost_exp_words):
                    got = dev.read_128b(1, hpost_base + i)
                    if got != exp:
                        hpost_mismatch += 1
                y_mismatch = 0
                for i, exp in enumerate(y_exp_words):
                    got = dev.read_128b(0, i)
                    if got != exp:
                        y_mismatch += 1
                assert hpost_mismatch == 0 and y_mismatch == 0, (
                    f"{w['name']} mismatch: hpost={hpost_mismatch}, y={y_mismatch}"
                )

                p1 = predict_layer_rtl_t8(w["l1_base"], w["l1_meta"], eng_mhz=clks["eng_mhz"], nom_sdram_mhz=clks["sdram_mhz"])
                p2 = predict_layer_rtl_t8(w["l2_base"], w["l2_meta"], eng_mhz=clks["eng_mhz"], nom_sdram_mhz=clks["sdram_mhz"])
                pred_tot = p1["pred_total_cyc"] + p2["pred_total_cyc"]
                pred_stall = p1["pred_stall_cyc"] + p2["pred_stall_cyc"]

                tot_cyc = int(c1["total_cycles"] + c2["total_cycles"])
                mac_cyc = int(c1["mac_cycles"] + c2["mac_cycles"])
                stall_cyc = int(c1["stall_cycles"] + c2["stall_cycles"])
                dma_cyc = int(c1["dma_cycles"] + c2["dma_cycles"])
                stall_pct = round(100.0 * stall_cyc / tot_cyc, 2)
                mac_util_pct = round(100.0 * mac_cyc / tot_cyc, 2)
                total_macs = 2 * T_TILE * d_model * d_ff
                time_ms = tot_cyc / (clks["eng_mhz"] * 1e3)
                gmac_s = total_macs / (time_ms * 1e6)
                err_pct = round(100.0 * (pred_tot - tot_cyc) / tot_cyc, 3)

                row = {
                    "name": w["name"],
                    "label": w["label"],
                    "d_model": d_model,
                    "d_ff": d_ff,
                    "t_tile": T_TILE,
                    "hpost_words": len(hpost_exp_words),
                    "y_words": len(y_exp_words),
                    "hpost_mismatch": hpost_mismatch,
                    "y_mismatch": y_mismatch,
                    "l1_total_cyc": int(c1["total_cycles"]),
                    "l1_stall_cyc": int(c1["stall_cycles"]),
                    "l1_dma_cyc": int(c1["dma_cycles"]),
                    "l1_mac_cyc": int(c1["mac_cycles"]),
                    "l2_total_cyc": int(c2["total_cycles"]),
                    "l2_stall_cyc": int(c2["stall_cycles"]),
                    "l2_dma_cyc": int(c2["dma_cycles"]),
                    "l2_mac_cyc": int(c2["mac_cycles"]),
                    "total_cycles": tot_cyc,
                    "mac_cycles": mac_cyc,
                    "stall_cycles": stall_cyc,
                    "dma_cycles": dma_cyc,
                    "stall_pct": stall_pct,
                    "mac_util_pct": mac_util_pct,
                    "time_ms": round(time_ms, 4),
                    "gmac_s": round(gmac_s, 4),
                    "pred_total_cyc": pred_tot,
                    "pred_stall_cyc": pred_stall,
                    "pred_stall_pct": round(100.0 * pred_stall / pred_tot, 2),
                    "pred_err_pct": err_pct,
                }
                shape_results.append(row)
                print(
                    f"[{w['name']}] 0-LSB PASS | tot={tot_cyc} (pred={pred_tot}, err={err_pct:+.3f}%) | "
                    f"stall={stall_cyc} ({stall_pct:.2f}%) | util={mac_util_pct:.2f}% | "
                    f"time={time_ms:.4f} ms | {gmac_s:.3f} GMAC/s"
                )

            payload = {
                "device": "EP4CE10E22C8N",
                "svf": str(TOP_T8_SVF),
                "t_tile": T_TILE,
                "board_id": id_info,
                "clocks": clks,
                "bist_1mib": bist,
                "shapes": shape_results,
            }
            OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Saved: {OUT_JSON}")
        finally:
            dev.close()
    finally:
        if TOP_T4_SVF.exists():
            flash_svf(TOP_T4_SVF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
