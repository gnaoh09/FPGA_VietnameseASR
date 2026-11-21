#!/usr/bin/env python3
"""Out-of-Sample Predictive Stall Validation on Unseen Real Zipformer-30M FFN Shapes
across {60, 90, 100, 120} MHz SDRAM clocks (zero fitted parameters).

Predictor parameters are derived solely from RTL state-machine schedules
(rtl/zipformer_engine.v, rtl/sdram_ctrl.v) and verified against independent
1 MiB OP_BIST page-burst reads — never fit to 512->1536->512 or any target FFN run.
"""

from __future__ import annotations

import json
import math
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
    pack_cfg4,
    pack_layer_w4_sdram_words,
    run_sdram_stream_layer,
    write_lut_256,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = ROOT_DIR / "build" / "output_files"

T_TILE = 4
RD_LAT = 4

FREQ_PLAN = [
    {"nom_sdram_mhz": 60, "svf": OF_DIR / "top_sdram60.svf", "ref_interval": 840, "t_rfc": 8},
    {"nom_sdram_mhz": 90, "svf": OF_DIR / "top_sdram90.svf", "ref_interval": 1260, "t_rfc": 8},
    {"nom_sdram_mhz": 100, "svf": OF_DIR / "top.svf", "ref_interval": 1400, "t_rfc": 8},
    {"nom_sdram_mhz": 120, "svf": OF_DIR / "top_sdram120.svf", "ref_interval": 1680, "t_rfc": 9},
]

UNSEEN_SHAPES = [
    {"name": "ffn_192_384_192", "label": "192->384->192 (stage_0)", "d_model": 192, "d_ff": 384},
    {"name": "ffn_256_576_256", "label": "256->576->256 (stage_1..5 FFN1)", "d_model": 256, "d_ff": 576},
    {"name": "ffn_256_960_256", "label": "256->960->256 (stage_1..5 FFN3)", "d_model": 256, "d_ff": 960},
]


def predict_layer_rtl(
    base_w128: int,
    layer_meta: dict,
    nom_sdram_mhz: float,
    ref_interval: int,
    t_rfc: int,
    eng_mhz: float = 80.0,
) -> dict:
    """Pure RTL-derived closed-form schedule predictor (locked pre-measurement)."""
    n_tiles = int(layer_meta["n_tiles"])
    k_chunks = int(layer_meta["k_chunks"])
    k_chunk = int(layer_meta["k_chunk"])
    pkt_words = int(layer_meta["pkt_words"])
    num_chunks = n_tiles * k_chunks

    c_tiles: list[int] = []
    c_dmas: list[float] = []
    pages_list: list[int] = []
    addr = base_w128
    for _nt in range(n_tiles):
        for kc in range(k_chunks):
            has_epi = (kc == k_chunks - 1)
            c_tile = 4 * k_chunk + (13 if has_epi else 6)
            pages = ((addr + pkt_words - 1) // 32) - (addr // 32) + 1
            sd_cyc = (8 * pkt_words + 14 * pages + 1) * (1.0 + t_rfc / float(ref_interval)) + 3.0
            c_dma = sd_cyc * (eng_mhz / nom_sdram_mhz) + 4.0
            c_tiles.append(c_tile)
            c_dmas.append(c_dma)
            pages_list.append(pages)
            addr += pkt_words

    stall = c_dmas[0]
    for idx in range(1, num_chunks):
        stall += max(0.0, c_dmas[idx] - c_tiles[idx - 1])
    active = float(sum(c_tiles))
    total = active + stall

    # Analytical critical SDRAM frequency where mean C_dma == mean C_tile
    mean_c_tile = active / num_chunks
    mean_pages = sum(pages_list) / num_chunks
    mean_sd_cyc = (8 * pkt_words + 14 * mean_pages + 1) * (1.0 + t_rfc / float(ref_interval)) + 3.0
    f_crit_mhz = (mean_sd_cyc * eng_mhz) / max(mean_c_tile - 4.0, 1.0)

    return {
        "pred_active_cyc": int(round(active)),
        "pred_dma_cyc": int(round(sum(c_dmas))),
        "pred_stall_cyc": int(round(stall)),
        "pred_total_cyc": int(round(total)),
        "pred_stall_pct": round(100.0 * stall / total, 2),
        "f_crit_mhz": round(f_crit_mhz, 2),
        "predicted_regime": "bandwidth_bound" if nom_sdram_mhz < f_crit_mhz else "compute_bound",
    }


def make_synthetic_integer_ffn(d_model: int, d_ff: int, seed: int) -> dict:
    """Create deterministic INT4/INT8 FFN workload matching RTL arithmetic range."""
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


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    # Build all 3 unseen workloads and contiguous SDRAM layout deterministically BEFORE running FPGA
    workloads = []
    all_sdram_words: list[int] = []
    for idx, spec in enumerate(UNSEEN_SHAPES):
        wl = make_synthetic_integer_ffn(spec["d_model"], spec["d_ff"], seed=20261119 + idx)
        l1_base = len(all_sdram_words)
        all_sdram_words.extend(wl["l1_words"])
        l2_base = len(all_sdram_words)
        all_sdram_words.extend(wl["l2_words"])
        workloads.append({
            **spec,
            **wl,
            "l1_base": l1_base,
            "l2_base": l2_base,
        })

    # Compute and print ALL locked predictions prior to touching hardware
    locked_preds: dict[str, dict] = {}
    print("=== LOCKED OUT-OF-SAMPLE RTL PREDICTIONS (NOMINAL 80 MHz ENG) ===")
    for fcfg in FREQ_PLAN:
        mhz = fcfg["nom_sdram_mhz"]
        f_key = f"sdram_{mhz}mhz"
        locked_preds[f_key] = {}
        for w in workloads:
            p_l1 = predict_layer_rtl(w["l1_base"], w["l1_meta"], mhz, fcfg["ref_interval"], fcfg["t_rfc"])
            p_l2 = predict_layer_rtl(w["l2_base"], w["l2_meta"], mhz, fcfg["ref_interval"], fcfg["t_rfc"])
            pred_tot = p_l1["pred_total_cyc"] + p_l2["pred_total_cyc"]
            pred_stall = p_l1["pred_stall_cyc"] + p_l2["pred_stall_cyc"]
            pred_pct = round(100.0 * pred_stall / pred_tot, 2)
            locked_preds[f_key][w["name"]] = {
                "l1": p_l1,
                "l2": p_l2,
                "pred_total_cyc": pred_tot,
                "pred_stall_cyc": pred_stall,
                "pred_stall_pct": pred_pct,
                "f_crit_l1_mhz": p_l1["f_crit_mhz"],
                "f_crit_l2_mhz": p_l2["f_crit_mhz"],
            }
            print(
                f"  @{mhz:3d}MHz {w['name']}: "
                f"L1(fcrit={p_l1['f_crit_mhz']}, st={p_l1['pred_stall_pct']}%) + "
                f"L2(fcrit={p_l2['f_crit_mhz']}, st={p_l2['pred_stall_pct']}%) -> "
                f"tot={pred_tot}, stall={pred_stall} ({pred_pct}%)"
            )

    results: dict = {
        "experiment": "Q2_Priority_2_Out_Of_Sample_Stall_Validation",
        "model_notes": "Locked purely from RTL schedules (rtl/zipformer_engine.v & rtl/sdram_ctrl.v); 0 parameters fit from 512->1536->512 or target runs.",
        "sdram_upload_bytes": len(all_sdram_words) * 16,
        "locked_nominal_predictions": locked_preds,
        "per_frequency": {},
    }

    try:
        for fcfg in FREQ_PLAN:
            mhz = fcfg["nom_sdram_mhz"]
            f_key = f"sdram_{mhz}mhz"
            flash_svf(fcfg["svf"])
            with UsbBlasterJtag() as dev:
                id_reg = dev.read_resp(0x2000)
                magic = (id_reg >> 32) & 0xFFFFFFFF
                assert magic == 0x5A463445
                eng_mhz, sd_mhz, _, _ = measure_pll_clocks(dev)
                dev.send_cmd(OP_WRITE_REG, 4, pack_cfg4(0, 0, 0, RD_LAT, 0))

                # Independent 1 MiB OP_BIST verification
                bist_1m = run_bist_stats(dev, 65536, sd_mhz)
                assert bist_1m["err_cnt"] == 0, f"1 MiB BIST error at {mhz} MHz"

                # Upload 3 unseen workloads ONCE per frequency
                up_s = dev.sdram_write_bulk_words(0, all_sdram_words)

                shape_entries = {}
                for w in workloads:
                    d_m, d_ff = w["d_model"], w["d_ff"]
                    x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
                    for idx_w, xw in enumerate(x_words):
                        dev.queue_write_128b(OP_COMMIT_ACT, idx_w, xw)
                    dev.flush()
                    write_lut_256(dev, w["lut"], lut_bank=0)

                    c_l1 = run_sdram_stream_layer(
                        dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                        dim_t=T_TILE, k_dim=d_m, n_dim=d_ff, layer_meta=w["l1_meta"],
                        sdram_base_w128=w["l1_base"], rd_lat=RD_LAT, lut_mode=1,
                        act_base=0, dst_base=128, dst_is_act=1,
                    )
                    hp_gold = [int(s, 16) for s in pack_words_128bit_i8(w["h_post_q"])]
                    hp_hw = [dev.read_128b(1, 128 + i) for i in range(len(hp_gold))]
                    hp_mism = sum(1 for a, b in zip(hp_hw, hp_gold) if a != b)
                    assert hp_mism == 0, f"{w['name']} @{mhz} L1 h_post mismatch={hp_mism}"

                    c_l2 = run_sdram_stream_layer(
                        dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                        dim_t=T_TILE, k_dim=d_ff, n_dim=d_m, layer_meta=w["l2_meta"],
                        sdram_base_w128=w["l2_base"], rd_lat=RD_LAT, lut_mode=0,
                        act_base=128, dst_base=0, dst_is_act=0,
                    )
                    y_gold = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]
                    y_hw = [dev.read_128b(0, i) for i in range(len(y_gold))]
                    y_mism = sum(1 for a, b in zip(y_hw, y_gold) if a != b)
                    assert y_mism == 0, f"{w['name']} @{mhz} L2 y_out mismatch={y_mism}"

                    # Also record locked prediction using exact measured PLL ratio
                    p_l1_nom = locked_preds[f_key][w["name"]]["l1"]
                    p_l2_nom = locked_preds[f_key][w["name"]]["l2"]
                    p_tot_nom = locked_preds[f_key][w["name"]]["pred_total_cyc"]
                    p_st_nom = locked_preds[f_key][w["name"]]["pred_stall_cyc"]
                    p_pct_nom = locked_preds[f_key][w["name"]]["pred_stall_pct"]

                    m_tot = c_l1["total_cycles"] + c_l2["total_cycles"]
                    m_st = c_l1["stall_cycles"] + c_l2["stall_cycles"]
                    m_pct = round(100.0 * m_st / m_tot, 2)
                    l1_pct = round(100.0 * c_l1["stall_cycles"] / c_l1["total_cycles"], 2)
                    l2_pct = round(100.0 * c_l2["stall_cycles"] / c_l2["total_cycles"], 2)

                    shape_entries[w["name"]] = {
                        "shape": w["label"],
                        "zero_lsb_hpost_bytes": len(hp_gold) * 16,
                        "zero_lsb_yout_bytes": len(y_gold) * 16,
                        "l1": {
                            "meas_total_cyc": c_l1["total_cycles"],
                            "meas_stall_cyc": c_l1["stall_cycles"],
                            "meas_stall_pct": l1_pct,
                            "pred_total_cyc": p_l1_nom["pred_total_cyc"],
                            "pred_stall_cyc": p_l1_nom["pred_stall_cyc"],
                            "pred_stall_pct": p_l1_nom["pred_stall_pct"],
                            "f_crit_pred_mhz": p_l1_nom["f_crit_mhz"],
                            "stall_err_pp": round(p_l1_nom["pred_stall_pct"] - l1_pct, 2),
                        },
                        "l2": {
                            "meas_total_cyc": c_l2["total_cycles"],
                            "meas_stall_cyc": c_l2["stall_cycles"],
                            "meas_stall_pct": l2_pct,
                            "pred_total_cyc": p_l2_nom["pred_total_cyc"],
                            "pred_stall_cyc": p_l2_nom["pred_stall_cyc"],
                            "pred_stall_pct": p_l2_nom["pred_stall_pct"],
                            "f_crit_pred_mhz": p_l2_nom["f_crit_mhz"],
                            "stall_err_pp": round(p_l2_nom["pred_stall_pct"] - l2_pct, 2),
                        },
                        "combined": {
                            "meas_total_cyc": m_tot,
                            "meas_stall_cyc": m_st,
                            "meas_stall_pct": m_pct,
                            "pred_total_cyc": p_tot_nom,
                            "pred_stall_cyc": p_st_nom,
                            "pred_stall_pct": p_pct_nom,
                            "total_cyc_err": p_tot_nom - m_tot,
                            "total_cyc_err_pct": round(100.0 * (p_tot_nom - m_tot) / m_tot, 2),
                            "stall_cyc_err": p_st_nom - m_st,
                            "stall_err_pp": round(p_pct_nom - m_pct, 2),
                        },
                    }
                    print(
                        f"  [MEAS @{mhz:3d}MHz {w['name']}] tot={m_tot} vs pred={p_tot_nom} "
                        f"({shape_entries[w['name']]['combined']['total_cyc_err_pct']:+.2f}%), "
                        f"stall={m_pct:.2f}% vs pred={p_pct_nom:.2f}% "
                        f"({shape_entries[w['name']]['combined']['stall_err_pp']:+.2f} pp) | 0 LSB PASS"
                    )

                results["per_frequency"][f_key] = {
                    "target_sdram_mhz": mhz,
                    "measured_eng_mhz": eng_mhz,
                    "measured_sdram_mhz": sd_mhz,
                    "bist_1mib_rd_MBps": bist_1m["rd_MBps_1e6"],
                    "bist_1mib_rd_pct": bist_1m["rd_pct_peak"],
                    "upload_s": round(up_s, 2),
                    "shapes": shape_entries,
                }
    finally:
        # Always restore canonical 100 MHz bitstream
        flash_svf(OF_DIR / "top.svf")

    # Summary error statistics across all 12 (frequency x FFN) pairs and 24 individual layers
    tot_errs = [
        abs(v["combined"]["total_cyc_err_pct"])
        for fdata in results["per_frequency"].values()
        for v in fdata["shapes"].values()
    ]
    st_errs = [
        abs(v["combined"]["stall_err_pp"])
        for fdata in results["per_frequency"].values()
        for v in fdata["shapes"].values()
    ]
    layer_st_errs = [
        abs(v[lyr]["stall_err_pp"])
        for fdata in results["per_frequency"].values()
        for v in fdata["shapes"].values()
        for lyr in ("l1", "l2")
    ]
    results["summary_errors"] = {
        "n_ffn_points": len(tot_errs),
        "n_layer_points": len(layer_st_errs),
        "max_abs_total_cyc_err_pct": round(max(tot_errs), 2),
        "mean_abs_total_cyc_err_pct": round(sum(tot_errs) / len(tot_errs), 2),
        "max_abs_ffn_stall_err_pp": round(max(st_errs), 2),
        "mean_abs_ffn_stall_err_pp": round(sum(st_errs) / len(st_errs), 2),
        "max_abs_layer_stall_err_pp": round(max(layer_st_errs), 2),
    }

    out_path = BUILD_DIR / "oos_stall_validation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved OOS validation results to {out_path}")
    print(json.dumps(results["summary_errors"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
