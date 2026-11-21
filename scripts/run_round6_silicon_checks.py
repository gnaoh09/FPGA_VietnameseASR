#!/usr/bin/env python3
"""Round-6 silicon checks (Section 4 items 2 and 4):
  4.2b: Record the 10 raw repeat cycle counts for 256->576->256 at T_tile=8, 80/80 MHz (top_t8_sd80.svf).
  4.4:  Verify RTL equivalence at CLK1_PHASE_DEG=0 by comparing freshly compiled top_equiv_p0.svf
        (built from current rtl/top.v and rtl/pll_80m.v at commit 8d4ce0c) against original top.svf
        across 3 FFN shapes (192->384->192, 192->640->192, 256->960->256) at 80/100 MHz (n=10, 0-LSB).
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
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from sdram_helpers import run_sdram_stream_layer, write_lut_256

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
OF_DIR = BUILD / "output_files"
SVF_DEFAULT = OF_DIR / "top.svf"
OUT_JSON = BUILD / "round6_silicon_checks.json"


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def run_wls_on_current_bitstream(wls: list[dict], sdram_words: list[int], t_tile: int) -> list[dict]:
    out = []
    with UsbBlasterJtag() as dev:
        dev.sdram_write_bulk_words(0, sdram_words)
        for w in wls:
            dm, df = w["d_model"], w["d_ff"]
            x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
            hpost_base = len(x_words)
            for i, xw in enumerate(x_words):
                dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
            dev.flush()
            write_lut_256(dev, w["lut"], lut_bank=0)
            raw_tots = []
            raw_stalls = []
            for rep in range(10):
                c1 = run_sdram_stream_layer(
                    dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                    t_tile, dm, df, w["l1_meta"], w["l1_base"], 4, 1, 0, hpost_base, 1,
                )
                c2 = run_sdram_stream_layer(
                    dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                    t_tile, df, dm, w["l2_meta"], w["l2_base"], 4, 0, hpost_base, 0, 0,
                )
                if rep == 0:
                    y_exp = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]
                    assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"0-LSB fail {w['shape']}"
                raw_tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                raw_stalls.append(int(c1["stall_cycles"] + c2["stall_cycles"]))
            out.append({
                "shape": w["shape"],
                "t_tile": t_tile,
                "raw_total_cycles_10": raw_tots,
                "raw_stall_cycles_10": raw_stalls,
                "mean_total": round(float(np.mean(raw_tots)), 2),
                "sd_total": round(float(np.std(raw_tots, ddof=1)), 2),
                "mean_stall": round(float(np.mean(raw_stalls)), 2),
            })
    return out


def main() -> int:
    # Prepare 3 T=4 shapes for Exp 4.4
    shapes_t4 = [(192, 384), (192, 640), (256, 960)]
    wls_t4 = []
    sdram_t4: list[int] = []
    for idx, (dm, df) in enumerate(shapes_t4):
        wl = make_ffn(dm, df, 4, seed=20261130 + idx)
        b1 = len(sdram_t4); sdram_t4.extend(wl["l1_words"])
        b2 = len(sdram_t4); sdram_t4.extend(wl["l2_words"])
        wls_t4.append({**wl, "shape": f"{dm}->{df}->{dm}", "l1_base": b1, "l2_base": b2})

    # Prepare 256->576->256 at T=8 for Exp 4.2b
    wl_t8 = make_ffn(256, 576, 8, seed=20261180)
    sdram_t8: list[int] = []
    b1_t8 = len(sdram_t8); sdram_t8.extend(wl_t8["l1_words"])
    b2_t8 = len(sdram_t8); sdram_t8.extend(wl_t8["l2_words"])
    wls_t8 = [{**wl_t8, "shape": "256->576->256", "l1_base": b1_t8, "l2_base": b2_t8}]

    res: dict = {}
    try:
        # 1. Exp 4.2b: T=8 80/80 MHz raw 10 repeats on 256->576->256
        flash_svf(OF_DIR / "top_t8_sd80.svf")
        t8_res = run_wls_on_current_bitstream(wls_t8, sdram_t8, 8)[0]
        res["exp42b_t8_256_576_80mhz_raw10"] = t8_res
        print(f"[Exp 4.2b T=8 256->576 @ 80/80] raw={t8_res['raw_total_cycles_10']} mean={t8_res['mean_total']}±{t8_res['sd_total']}")

        # 2. Exp 4.4: Fresh top_equiv_p0.svf vs original top.svf
        flash_svf(OF_DIR / "top_equiv_p0.svf")
        new_res = run_wls_on_current_bitstream(wls_t4, sdram_t4, 4)

        flash_svf(SVF_DEFAULT)
        orig_res = run_wls_on_current_bitstream(wls_t4, sdram_t4, 4)

        comp = []
        for nr, orr in zip(new_res, orig_res):
            diff = nr["mean_total"] - orr["mean_total"]
            comp.append({
                "shape": nr["shape"],
                "orig_top_mean": orr["mean_total"],
                "orig_top_sd": orr["sd_total"],
                "new_equiv_p0_mean": nr["mean_total"],
                "new_equiv_p0_sd": nr["sd_total"],
                "mean_diff_cyc": round(diff, 2),
                "orig_raw10": orr["raw_total_cycles_10"],
                "new_raw10": nr["raw_total_cycles_10"],
            })
            print(
                f"[Exp 4.4 {nr['shape']:13s}] orig_top={orr['mean_total']:.1f}±{orr['sd_total']:.2f} "
                f"vs new_equiv_p0={nr['mean_total']:.1f}±{nr['sd_total']:.2f} (diff={diff:+.2f} cyc)"
            )
        res["exp44_rtl_equivalence_check"] = comp
    finally:
        if SVF_DEFAULT.exists():
            flash_svf(SVF_DEFAULT)

    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
