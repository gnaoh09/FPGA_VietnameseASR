#!/usr/bin/env python3
"""Controlled 2x2 region-swapped silicon experiment on EP4CE10E22C8N (engine 80 MHz / SDRAM 90 MHz):
  192->640->192 FFN, T_tile=4, W4A8, offsets {0, 1, 15, 31}.

Controls for SDRAM region placement by executing three blocks in one continuous session:
  1) layout_a_orig (N=20 counterbalanced rounds):
       offset 0 @ 0, offset 1 @ 16385, offset 15 @ 32783, offset 31 @ 49183
  2) layout_b_swapped (N=20 counterbalanced rounds):
       offset 0 @ 0, offset 15 @ 16399, offset 1 @ 32769, offset 31 @ 49183
  3) same_region_low_addr (N=10 counterbalanced pairs):
       offset 1 @ 1 vs offset 15 @ 15 (overwritten identically at low memory)

Leaves existing build/offset_192_640_90mhz_results.json untouched, saves new results to
build/offset_192_640_90mhz_region_swap_results.json, and always restores canonical top.svf.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_oos_12pt_ablation import predict_layer_ablation
from fpga_jtag_runner import OP_WRITE_REG, UsbBlasterJtag
from gen_golden_vectors import pack_words_128bit_i8
from run_g4b_sdram_experiment import measure_pll_clocks, run_bist_stats
from run_offset_192_640_90mhz import (
    BUILD_DIR,
    OF_DIR,
    OFFSETS,
    RD_LAT,
    STRIDE_REGION_W128,
    T_TILE,
    flash_svf,
    run_one_ffn,
    summarize_vec,
)
from run_oos_stall_validation import make_synthetic_integer_ffn
from sdram_helpers import pack_cfg4, write_lut_256

N_ROUNDS_LAYOUT = 20
N_SAME_REGION_PAIRS = 10

# Exact two-sided 95% t critical values (alpha = 0.05) for df = n - 1
T_CRIT_95 = {
    9: 2.2621571627409915,
    19: 2.093024054408263,
    39: 2.022690911734728,
}


def compute_predictions(wl: dict, eng_mhz: float, sd_mhz: float) -> list[dict]:
    """Compute nominal (80/90) and measured-PLL per-packet & mean-pages predictions."""
    l1_len = len(wl["l1_words"])
    rows = []
    for off in OFFSETS:
        l1_base = off
        l2_base = off + l1_len

        nom_l1 = predict_layer_ablation(l1_base, wl["l1_meta"], 90.0, 1260, 8, eng_mhz=80.0, use_mean_pages=False)
        nom_l2 = predict_layer_ablation(l2_base, wl["l2_meta"], 90.0, 1260, 8, eng_mhz=80.0, use_mean_pages=False)
        nom_m1 = predict_layer_ablation(l1_base, wl["l1_meta"], 90.0, 1260, 8, eng_mhz=80.0, use_mean_pages=True)
        nom_m2 = predict_layer_ablation(l2_base, wl["l2_meta"], 90.0, 1260, 8, eng_mhz=80.0, use_mean_pages=True)

        pll_l1 = predict_layer_ablation(l1_base, wl["l1_meta"], sd_mhz, 1260, 8, eng_mhz=eng_mhz, use_mean_pages=False)
        pll_l2 = predict_layer_ablation(l2_base, wl["l2_meta"], sd_mhz, 1260, 8, eng_mhz=eng_mhz, use_mean_pages=False)
        pll_m1 = predict_layer_ablation(l1_base, wl["l1_meta"], sd_mhz, 1260, 8, eng_mhz=eng_mhz, use_mean_pages=True)
        pll_m2 = predict_layer_ablation(l2_base, wl["l2_meta"], sd_mhz, 1260, 8, eng_mhz=eng_mhz, use_mean_pages=True)

        rows.append(
            {
                "offset_w128": off,
                "nominal_80_90": {
                    "per_packet": {"l1": nom_l1, "l2": nom_l2, "combined": nom_l1 + nom_l2},
                    "mean_pages": {"l1": nom_m1, "l2": nom_m2, "combined": nom_m1 + nom_m2},
                },
                "measured_pll": {
                    "l1_per_packet": pll_l1,
                    "l2_per_packet": pll_l2,
                    "combined_per_packet": pll_l1 + pll_l2,
                    "l1_mean_pages": pll_m1,
                    "l2_mean_pages": pll_m2,
                    "combined_mean_pages": pll_m1 + pll_m2,
                },
            }
        )
    return rows


def paired_stats_block(records: list[dict]) -> dict:
    """Compute comprehensive paired statistics (offset 1 - offset 15) for a block of records."""
    d_comb = [r["runs"]["1"]["combined_total_cyc"] - r["runs"]["15"]["combined_total_cyc"] for r in records]
    d_l1 = [r["runs"]["1"]["l1_total_cyc"] - r["runs"]["15"]["l1_total_cyc"] for r in records]
    d_l2 = [r["runs"]["1"]["l2_total_cyc"] - r["runs"]["15"]["l2_total_cyc"] for r in records]

    d_1_first = [
        r["runs"]["1"]["combined_total_cyc"] - r["runs"]["15"]["combined_total_cyc"]
        for r in records
        if r["order"].index(1) < r["order"].index(15)
    ]
    d_15_first = [
        r["runs"]["1"]["combined_total_cyc"] - r["runs"]["15"]["combined_total_cyc"]
        for r in records
        if r["order"].index(15) < r["order"].index(1)
    ]

    n = len(d_comb)
    df = n - 1
    assert df in T_CRIT_95, f"Add t_crit for df={df}"
    t_crit = T_CRIT_95[df]

    mean_c = statistics.mean(d_comb)
    sd_c = statistics.stdev(d_comb)
    se_c = sd_c / math.sqrt(n)
    ci_lo = mean_c - t_crit * se_c
    ci_hi = mean_c + t_crit * se_c

    def sub_summary(vals: list[int]) -> dict:
        s = summarize_vec(vals)
        se_v = s["sd"] / math.sqrt(len(vals))
        return {**s, "se": round(se_v, 3), "positive_count": sum(1 for v in vals if v > 0)}

    return {
        "n": n,
        "df": df,
        "t_crit_95": round(t_crit, 4),
        "combined": {
            **summarize_vec(d_comb),
            "se": round(se_c, 3),
            "ci95_t": [round(ci_lo, 2), round(ci_hi, 2)],
            "positive_count": sum(1 for v in d_comb if v > 0),
            "t_vs_0": round(mean_c / se_c, 2),
            "t_vs_42": round((mean_c - 42.0) / se_c, 2),
        },
        "l1": sub_summary(d_l1),
        "l2": sub_summary(d_l2),
        "order_split": {
            "order_1_before_15_n": len(d_1_first),
            "order_1_before_15_mean": round(statistics.mean(d_1_first), 3) if d_1_first else None,
            "order_15_before_1_n": len(d_15_first),
            "order_15_before_1_mean": round(statistics.mean(d_15_first), 3) if d_15_first else None,
        },
    }


def summarize_per_offset(records: list[dict], offsets: list[int], region_bases: dict[int, int], l1_len: int) -> dict:
    """Summarize combined, L1, L2, and stall cycles per offset in a block."""
    out = {}
    for off in offsets:
        vals = [r["runs"][str(off)] for r in records]
        out[str(off)] = {
            "base_l1_w128": region_bases[off],
            "base_l2_w128": region_bases[off] + l1_len,
            "combined_total_cyc": summarize_vec([v["combined_total_cyc"] for v in vals]),
            "l1_total_cyc": summarize_vec([v["l1_total_cyc"] for v in vals]),
            "l2_total_cyc": summarize_vec([v["l2_total_cyc"] for v in vals]),
            "combined_stall_cyc": summarize_vec([v["combined_stall_cyc"] for v in vals]),
        }
    return out


def run_layout_block(
    dev: UsbBlasterJtag,
    name: str,
    region_bases: dict[int, int],
    wl: dict,
    combined_words: list[int],
    x_words: list[int],
    hp_gold: list[int],
    y_gold: list[int],
    n_rounds: int,
) -> tuple[dict, list[dict]]:
    """Upload 4 copies at region_bases and execute n_rounds counterbalanced interleaved trials."""
    l1_len = len(wl["l1_words"])
    upload_times = {}
    for off, l1_b in region_bases.items():
        up_s = dev.sdram_write_bulk_words(l1_b, combined_words)
        upload_times[str(off)] = round(up_s, 3)
    print(f"\n[{name}] Uploaded 4 copies ({region_bases}): upload_s={upload_times}")

    rounds_log: list[dict] = []
    for r_idx in range(n_rounds):
        order = [1, 15, 0, 31] if (r_idx % 2 == 0) else [15, 1, 31, 0]
        rec = {"round": r_idx, "layout": name, "order": order, "runs": {}}
        for off in order:
            l1_b = region_bases[off]
            l2_b = l1_b + l1_len
            res = run_one_ffn(dev, wl, l1_b, l2_b, x_words, hp_gold, y_gold)
            assert res["zero_lsb_verified"] is True
            rec["runs"][str(off)] = res
        d1_15 = rec["runs"]["1"]["combined_total_cyc"] - rec["runs"]["15"]["combined_total_cyc"]
        d_l1 = rec["runs"]["1"]["l1_total_cyc"] - rec["runs"]["15"]["l1_total_cyc"]
        d_l2 = rec["runs"]["1"]["l2_total_cyc"] - rec["runs"]["15"]["l2_total_cyc"]
        print(
            f"  [{name}] Round {r_idx:02d} {order}: "
            f"off0={rec['runs']['0']['combined_total_cyc']} off1={rec['runs']['1']['combined_total_cyc']} "
            f"off15={rec['runs']['15']['combined_total_cyc']} off31={rec['runs']['31']['combined_total_cyc']} | "
            f"d(1-15)={d1_15:+d} (L1={d_l1:+d}, L2={d_l2:+d})"
        )
        rounds_log.append(rec)
    return upload_times, rounds_log


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    wl = make_synthetic_integer_ffn(192, 640, seed=20261121)
    combined_words = wl["l1_words"] + wl["l2_words"]
    l1_len = len(wl["l1_words"])
    x_words = [int(s, 16) for s in pack_words_128bit_i8(wl["x_q"])]
    hp_gold = [int(s, 16) for s in pack_words_128bit_i8(wl["h_post_q"])]
    y_gold = [int(s, 16) for s in pack_words_128bit_i8(wl["y_q"])]

    region_bases_a = {
        0: 0 * STRIDE_REGION_W128 + 0,
        1: 1 * STRIDE_REGION_W128 + 1,
        15: 2 * STRIDE_REGION_W128 + 15,
        31: 3 * STRIDE_REGION_W128 + 31,
    }
    region_bases_b = {
        0: 0 * STRIDE_REGION_W128 + 0,
        15: 1 * STRIDE_REGION_W128 + 15,
        1: 2 * STRIDE_REGION_W128 + 1,
        31: 3 * STRIDE_REGION_W128 + 31,
    }
    region_bases_same = {1: 1, 15: 15}

    rounds_a: list[dict] = []
    rounds_b: list[dict] = []
    same_pairs: list[dict] = []
    up_a: dict = {}
    up_b: dict = {}

    try:
        flash_svf(OF_DIR / "top_sdram90.svf")
        with UsbBlasterJtag() as dev:
            id_reg = dev.read_resp(0x2000)
            magic = (id_reg >> 32) & 0xFFFFFFFF
            assert magic == 0x5A463445, f"Unexpected engine magic=0x{magic:08x}"

            eng_mhz, sd_mhz, _, _ = measure_pll_clocks(dev)
            dev.send_cmd(OP_WRITE_REG, 4, pack_cfg4(0, 0, 0, RD_LAT, 0))
            bist_1m = run_bist_stats(dev, 65536, sd_mhz)
            assert bist_1m["err_cnt"] == 0, f"1 MiB BIST failed: {bist_1m}"
            print(
                f"[90MHz] eng={eng_mhz:.3f} MHz, sdram={sd_mhz:.3f} MHz, "
                f"BIST=0 errs ({bist_1m['rd_MBps_1e6']:.2f} MB/s)"
            )

            write_lut_256(dev, wl["lut"], lut_bank=0)

            up_a, rounds_a = run_layout_block(
                dev, "layout_a_orig", region_bases_a, wl, combined_words,
                x_words, hp_gold, y_gold, N_ROUNDS_LAYOUT,
            )
            up_b, rounds_b = run_layout_block(
                dev, "layout_b_swapped", region_bases_b, wl, combined_words,
                x_words, hp_gold, y_gold, N_ROUNDS_LAYOUT,
            )

            print(f"\n[same_region_low_addr] Running N={N_SAME_REGION_PAIRS} counterbalanced overwrite pairs (addr 1 vs 15)")
            for p_idx in range(N_SAME_REGION_PAIRS):
                seq = [1, 15] if (p_idx % 2 == 0) else [15, 1]
                pair = {"pair_idx": p_idx, "order": seq, "runs": {}}
                for off in seq:
                    dev.sdram_write_bulk_words(off, combined_words)
                    res = run_one_ffn(dev, wl, off, off + l1_len, x_words, hp_gold, y_gold)
                    assert res["zero_lsb_verified"] is True
                    pair["runs"][str(off)] = res
                d_tot = pair["runs"]["1"]["combined_total_cyc"] - pair["runs"]["15"]["combined_total_cyc"]
                d_l1 = pair["runs"]["1"]["l1_total_cyc"] - pair["runs"]["15"]["l1_total_cyc"]
                d_l2 = pair["runs"]["1"]["l2_total_cyc"] - pair["runs"]["15"]["l2_total_cyc"]
                print(
                    f"  [same_region] Pair {p_idx:02d} {seq}: "
                    f"off1={pair['runs']['1']['combined_total_cyc']} off15={pair['runs']['15']['combined_total_cyc']} | "
                    f"d(1-15)={d_tot:+d} (L1={d_l1:+d}, L2={d_l2:+d})"
                )
                same_pairs.append(pair)
    finally:
        flash_svf(OF_DIR / "top.svf")

    predictions = compute_predictions(wl, eng_mhz, sd_mhz)
    stats_a = paired_stats_block(rounds_a)
    stats_b = paired_stats_block(rounds_b)
    stats_pooled = paired_stats_block(rounds_a + rounds_b)
    stats_same = paired_stats_block(same_pairs)

    mean_a = stats_a["combined"]["mean"]
    mean_b = stats_b["combined"]["mean"]
    se_a = stats_a["combined"]["se"]
    se_b = stats_b["combined"]["se"]
    pure_offset_est = 0.5 * (mean_a + mean_b)
    region_bias_est = 0.5 * (mean_a - mean_b)
    crossover_se = 0.5 * math.sqrt(se_a ** 2 + se_b ** 2)

    all_zero_lsb = (
        all(r["runs"][str(o)]["zero_lsb_verified"] for r in (rounds_a + rounds_b) for o in OFFSETS)
        and all(p["runs"][str(o)]["zero_lsb_verified"] for p in same_pairs for o in (1, 15))
    )

    out = {
        "experiment": "Controlled_2x2_Region_Swap_Offset_Test_192_640_192_90MHz",
        "shape": "ffn_192_640_192",
        "t_tile": T_TILE,
        "seed": 20261121,
        "clocks_mhz": {
            "target_eng": 80.0,
            "target_sdram": 90.0,
            "measured_eng": eng_mhz,
            "measured_sdram": sd_mhz,
        },
        "bist_1mib": bist_1m,
        "all_runs_zero_lsb_verified": all_zero_lsb,
        "predictions": predictions,
        "layout_bases_w128": {
            "layout_a_orig": region_bases_a,
            "layout_b_swapped": region_bases_b,
            "same_region_low_addr": region_bases_same,
        },
        "upload_times_s": {
            "layout_a_orig": up_a,
            "layout_b_swapped": up_b,
        },
        "per_offset_stats": {
            "layout_a_orig": summarize_per_offset(rounds_a, OFFSETS, region_bases_a, l1_len),
            "layout_b_swapped": summarize_per_offset(rounds_b, OFFSETS, region_bases_b, l1_len),
            "same_region_low_addr": summarize_per_offset(same_pairs, [1, 15], region_bases_same, l1_len),
        },
        "paired_delta_offset1_minus_offset15": {
            "layout_a_orig": stats_a,
            "layout_b_swapped": stats_b,
            "pooled_ab": stats_pooled,
            "same_region_low_addr": stats_same,
        },
        "crossover_2x2_decomposition": {
            "formula_note": "pure_offset = 0.5*(mean_A + mean_B); region_bias_reg1_minus_reg2 = 0.5*(mean_A - mean_B)",
            "pure_offset_est_cyc": round(pure_offset_est, 3),
            "region_bias_reg1_minus_reg2_cyc": round(region_bias_est, 3),
            "crossover_se_cyc": round(crossover_se, 3),
            "pure_offset_ci95_approx": [
                round(pure_offset_est - T_CRIT_95[39] * crossover_se, 2),
                round(pure_offset_est + T_CRIT_95[39] * crossover_se, 2),
            ],
            "region_bias_ci95_approx": [
                round(region_bias_est - T_CRIT_95[39] * crossover_se, 2),
                round(region_bias_est + T_CRIT_95[39] * crossover_se, 2),
            ],
        },
        "rounds_layout_a_orig": rounds_a,
        "rounds_layout_b_swapped": rounds_b,
        "same_region_pairs": same_pairs,
    }

    out_path = BUILD_DIR / "offset_192_640_90mhz_region_swap_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\n=== SUMMARY: 2x2 REGION-SWAPPED OFFSET CONTROL (192->640->192 @ 80/90 MHz) ===")
    for label, st in [
        ("layout_a_orig (N=20)", stats_a),
        ("layout_b_swapped (N=20)", stats_b),
        ("pooled_ab (N=40)", stats_pooled),
        ("same_region_low_addr (N=10)", stats_same),
    ]:
        c = st["combined"]
        print(
            f"  {label:28s}: delta={c['mean']:+.2f}±{c['sd']:.2f} (SE={c['se']:.2f}, 95% t-CI {c['ci95_t']}, "
            f"med={c['median']:+.1f} [{c['min']}..{c['max']}], {c['positive_count']}/{st['n']} > 0, "
            f"t0={c['t_vs_0']}, t42={c['t_vs_42']}) | L1={st['l1']['mean']:+.2f}, L2={st['l2']['mean']:+.2f}"
        )
    cd = out["crossover_2x2_decomposition"]
    print(
        f"  Crossover 2x2: pure_offset={cd['pure_offset_est_cyc']:+.2f} {cd['pure_offset_ci95_approx']}, "
        f"region_bias(R1-R2)={cd['region_bias_reg1_minus_reg2_cyc']:+.2f} {cd['region_bias_ci95_approx']}"
    )
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
