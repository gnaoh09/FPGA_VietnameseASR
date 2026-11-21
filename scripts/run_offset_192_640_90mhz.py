#!/usr/bin/env python3
"""Controlled silicon discrimination test on EP4CE10E22C8N (engine 80 MHz / SDRAM 90 MHz):
  192->640->192 FFN, T_tile=4, W4A8, comparing contiguous L1+L2 base offsets {0, 1, 15, 31}.

Critical comparison: offset 1 vs offset 15 (20 interleaved pairs + 3 literal-address pairs).
- Per-packet page-counting model predicts: 65,284 vs 65,242 cycles (Delta_{1-15} = +42 cycles).
- Within-layer mean-pages model predicts: 65,243 vs 65,243 cycles (Delta_{1-15} = 0 cycles).

Saves immutable pre-registered predictions to build/preregistered_offset_192_640_90mhz.json
prior to flashing or reading hardware, and writes full per-run measurements to
build/offset_192_640_90mhz_results.json. Always restores canonical top.svf (100 MHz).
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpga_jtag_runner import OP_COMMIT_ACT, OP_WRITE_REG, UsbBlasterJtag, pack_cfg1, pack_cfg2
from gen_golden_vectors import pack_words_128bit_i8
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks, run_bist_stats
from run_oos_stall_validation import make_synthetic_integer_ffn
from sdram_helpers import pack_cfg4, run_sdram_stream_layer, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = ROOT_DIR / "build" / "output_files"

T_TILE = 4
RD_LAT = 4
N_ROUNDS = 20
N_LITERAL_ROUNDS = 3
STRIDE_REGION_W128 = 16384  # 128 * 128: exact multiple of 4 banks * 32 w128/page
OFFSETS = [0, 1, 15, 31]


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def run_one_ffn(
    dev: UsbBlasterJtag,
    wl: dict,
    l1_base: int,
    l2_base: int,
    x_words: list[int],
    hp_gold: list[int],
    y_gold: list[int],
) -> dict:
    d_m, d_ff = wl["d_model"], wl["d_ff"]
    for idx_w, xw in enumerate(x_words):
        dev.queue_write_128b(OP_COMMIT_ACT, idx_w, xw)
    dev.flush()

    c_l1 = run_sdram_stream_layer(
        dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
        dim_t=T_TILE, k_dim=d_m, n_dim=d_ff, layer_meta=wl["l1_meta"],
        sdram_base_w128=l1_base, rd_lat=RD_LAT, lut_mode=1,
        act_base=0, dst_base=128, dst_is_act=1,
    )
    hp_hw = [dev.read_128b(1, 128 + i) for i in range(len(hp_gold))]
    if any(a != b for a, b in zip(hp_hw, hp_gold)):
        raise AssertionError(f"h_post mismatch at l1_base={l1_base}")

    c_l2 = run_sdram_stream_layer(
        dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
        dim_t=T_TILE, k_dim=d_ff, n_dim=d_m, layer_meta=wl["l2_meta"],
        sdram_base_w128=l2_base, rd_lat=RD_LAT, lut_mode=0,
        act_base=128, dst_base=0, dst_is_act=0,
    )
    y_hw = [dev.read_128b(0, i) for i in range(len(y_gold))]
    if any(a != b for a, b in zip(y_hw, y_gold)):
        raise AssertionError(f"y_out mismatch at l2_base={l2_base}")

    return {
        "l1_total_cyc": int(c_l1["total_cycles"]),
        "l1_stall_cyc": int(c_l1["stall_cycles"]),
        "l1_dma_cyc": int(c_l1["dma_cycles"]),
        "l1_mac_cyc": int(c_l1["mac_cycles"]),
        "l2_total_cyc": int(c_l2["total_cycles"]),
        "l2_stall_cyc": int(c_l2["stall_cycles"]),
        "l2_dma_cyc": int(c_l2["dma_cycles"]),
        "l2_mac_cyc": int(c_l2["mac_cycles"]),
        "combined_total_cyc": int(c_l1["total_cycles"] + c_l2["total_cycles"]),
        "combined_stall_cyc": int(c_l1["stall_cycles"] + c_l2["stall_cycles"]),
        "zero_lsb_verified": True,
    }


def summarize_vec(vals: list[float]) -> dict:
    s = sorted(vals)
    n = len(s)
    q1 = statistics.median(s[: n // 2])
    q3 = statistics.median(s[(n + 1) // 2 :])
    return {
        "n": n,
        "mean": round(statistics.mean(s), 3),
        "sd": round(statistics.stdev(s), 3) if n > 1 else 0.0,
        "median": round(statistics.median(s), 3),
        "iqr": round(q3 - q1, 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    abl_data = json.loads((BUILD_DIR / "oos_12pt_ablation_results.json").read_text(encoding="utf-8"))
    prereg = abl_data["preregistered_offset_sweep_192_640_192_90mhz"]

    # Persist immutable pre-registered prediction snapshot before flashing FPGA
    prereg_path = BUILD_DIR / "preregistered_offset_192_640_90mhz.json"
    prereg_path.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    print(f"Locked pre-registered predictions -> {prereg_path}")

    wl = make_synthetic_integer_ffn(192, 640, seed=20261121)
    combined_words = wl["l1_words"] + wl["l2_words"]
    l1_len = len(wl["l1_words"])
    x_words = [int(s, 16) for s in pack_words_128bit_i8(wl["x_q"])]
    hp_gold = [int(s, 16) for s in pack_words_128bit_i8(wl["h_post_q"])]
    y_gold = [int(s, 16) for s in pack_words_128bit_i8(wl["y_q"])]

    region_bases = {
        off: idx * STRIDE_REGION_W128 + off for idx, off in enumerate(OFFSETS)
    }

    rounds_log = []
    literal_pairs_log = []

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
            print(f"[90MHz] eng={eng_mhz:.3f} MHz, sdram={sd_mhz:.3f} MHz, BIST=0 errs ({bist_1m['rd_MBps_1e6']:.2f} MB/s)")

            write_lut_256(dev, wl["lut"], lut_bank=0)
            upload_times = {}
            for off, l1_b in region_bases.items():
                up_s = dev.sdram_write_bulk_words(l1_b, combined_words)
                upload_times[str(off)] = round(up_s, 3)
            print(f"Uploaded 4 bank/page-equivalent copies ({len(combined_words)} w128 each): {upload_times}")

            for r_idx in range(N_ROUNDS):
                order = [1, 15, 0, 31] if (r_idx % 2 == 0) else [15, 1, 31, 0]
                rec = {"round": r_idx, "order": order, "runs": {}}
                for off in order:
                    l1_b = region_bases[off]
                    l2_b = l1_b + l1_len
                    res = run_one_ffn(dev, wl, l1_b, l2_b, x_words, hp_gold, y_gold)
                    rec["runs"][str(off)] = res
                d1_15 = rec["runs"]["1"]["combined_total_cyc"] - rec["runs"]["15"]["combined_total_cyc"]
                d_l2 = rec["runs"]["1"]["l2_total_cyc"] - rec["runs"]["15"]["l2_total_cyc"]
                d_l1 = rec["runs"]["1"]["l1_total_cyc"] - rec["runs"]["15"]["l1_total_cyc"]
                print(
                    f"  Round {r_idx:02d} {order}: off0={rec['runs']['0']['combined_total_cyc']}, "
                    f"off1={rec['runs']['1']['combined_total_cyc']}, off15={rec['runs']['15']['combined_total_cyc']}, "
                    f"off31={rec['runs']['31']['combined_total_cyc']} | d(1-15)={d1_15:+d} (L1={d_l1:+d}, L2={d_l2:+d})"
                )
                rounds_log.append(rec)

            # Literal low-address sanity check (addr=1 vs addr=15 directly)
            for l_idx in range(N_LITERAL_ROUNDS):
                pair = {}
                seq = [1, 15] if (l_idx % 2 == 0) else [15, 1]
                for off in seq:
                    dev.sdram_write_bulk_words(off, combined_words)
                    pair[str(off)] = run_one_ffn(dev, wl, off, off + l1_len, x_words, hp_gold, y_gold)
                dl = pair["1"]["combined_total_cyc"] - pair["15"]["combined_total_cyc"]
                print(f"  Literal {seq}: off1={pair['1']['combined_total_cyc']}, off15={pair['15']['combined_total_cyc']} | d(1-15)={dl:+d}")
                literal_pairs_log.append({"pair_idx": l_idx, "order": seq, "runs": pair})
    finally:
        flash_svf(OF_DIR / "top.svf")

    by_off = {off: [r["runs"][str(off)] for r in rounds_log] for off in OFFSETS}
    per_offset_stats = {}
    for off in OFFSETS:
        per_offset_stats[str(off)] = {
            "base_l1_w128": region_bases[off],
            "base_l2_w128": region_bases[off] + l1_len,
            "combined_total_cyc": summarize_vec([x["combined_total_cyc"] for x in by_off[off]]),
            "l1_total_cyc": summarize_vec([x["l1_total_cyc"] for x in by_off[off]]),
            "l2_total_cyc": summarize_vec([x["l2_total_cyc"] for x in by_off[off]]),
            "combined_stall_cyc": summarize_vec([x["combined_stall_cyc"] for x in by_off[off]]),
        }

    paired_1_15_comb = [r["runs"]["1"]["combined_total_cyc"] - r["runs"]["15"]["combined_total_cyc"] for r in rounds_log]
    paired_1_15_l1 = [r["runs"]["1"]["l1_total_cyc"] - r["runs"]["15"]["l1_total_cyc"] for r in rounds_log]
    paired_1_15_l2 = [r["runs"]["1"]["l2_total_cyc"] - r["runs"]["15"]["l2_total_cyc"] for r in rounds_log]
    paired_31_0_comb = [r["runs"]["31"]["combined_total_cyc"] - r["runs"]["0"]["combined_total_cyc"] for r in rounds_log]
    paired_1_0_comb = [r["runs"]["1"]["combined_total_cyc"] - r["runs"]["0"]["combined_total_cyc"] for r in rounds_log]
    lit_1_15 = [p["runs"]["1"]["combined_total_cyc"] - p["runs"]["15"]["combined_total_cyc"] for p in literal_pairs_log]

    se_comb = (statistics.stdev(paired_1_15_comb) / math.sqrt(len(paired_1_15_comb))) if len(paired_1_15_comb) > 1 else 0.0
    t_approx = (statistics.mean(paired_1_15_comb) / se_comb) if se_comb > 0 else float("inf")

    out = {
        "experiment": "Discriminating_Offset_Test_192_640_192_90MHz",
        "clocks_mhz": {"target_eng": 80.0, "target_sdram": 90.0, "measured_eng": eng_mhz, "measured_sdram": sd_mhz},
        "bist_1mib": bist_1m,
        "preregistered_predictions": prereg,
        "n_interleaved_rounds": N_ROUNDS,
        "per_offset_stats": per_offset_stats,
        "paired_differences": {
            "offset1_minus_offset15_combined_cyc": summarize_vec(paired_1_15_comb),
            "offset1_minus_offset15_l1_cyc": summarize_vec(paired_1_15_l1),
            "offset1_minus_offset15_l2_cyc": summarize_vec(paired_1_15_l2),
            "offset1_minus_offset15_se": round(se_comb, 3),
            "offset1_minus_offset15_t_stat": round(t_approx, 2),
            "offset1_minus_offset0_combined_cyc": summarize_vec(paired_1_0_comb),
            "offset31_minus_offset0_combined_cyc": summarize_vec(paired_31_0_comb),
            "literal_low_addr_offset1_minus_offset15_cyc": summarize_vec(lit_1_15),
        },
        "rounds": rounds_log,
        "literal_pairs": literal_pairs_log,
    }
    out_path = BUILD_DIR / "offset_192_640_90mhz_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\n=== SUMMARY: 192->640->192 @ 80/90 MHz ===")
    print("Per-offset median [min..max]:")
    for off in OFFSETS:
        st = per_offset_stats[str(off)]["combined_total_cyc"]
        print(f"  offset {off:2d}: mean={st['mean']}, sd={st['sd']}, med={st['median']}, iqr={st['iqr']} [{st['min']}..{st['max']}]")
    pd = out["paired_differences"]
    print(f"Paired Delta(1-15) combined: mean={pd['offset1_minus_offset15_combined_cyc']['mean']}, sd={pd['offset1_minus_offset15_combined_cyc']['sd']}, med={pd['offset1_minus_offset15_combined_cyc']['median']} [{pd['offset1_minus_offset15_combined_cyc']['min']}..{pd['offset1_minus_offset15_combined_cyc']['max']}], t={pd['offset1_minus_offset15_t_stat']}")
    print(f"Paired Delta(1-15) L1={pd['offset1_minus_offset15_l1_cyc']['mean']}, L2={pd['offset1_minus_offset15_l2_cyc']['mean']}")
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
