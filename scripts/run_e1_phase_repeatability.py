#!/usr/bin/env python3
"""E1 Static-Phase Repeatability Experiment on EP4CE10E22C8N:
Re-flashes 80/80 MHz (0 deg, top_sdram80.svf) 10 times and 90/90 MHz (0 deg,
top_eng90_sd90.svf) 10 times (alternating with top.svf / other bitstream or
direct SRAM reconfiguration + PLL re-lock), measuring the 3 unseen shapes
(128->384->128, 320->960->320, 384->1152->384; n=5 repeats/shape, 0-LSB verified)
on every flash to test whether 1:1 CDC slip is deterministic across re-locks
or varies stochastically from flash to flash.

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
from run_43_silicon_heldout import make_ffn, predict_rtl_gen
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from sdram_helpers import run_sdram_stream_layer, write_lut_256

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
OF_DIR = BUILD / "output_files"
SVF_DEFAULT = OF_DIR / "top.svf"
OUT_JSON = BUILD / "e1_phase_repeatability_results.json"


def flash_svf(svf_path: Path) -> None:
    subprocess.run(
        ["openFPGALoader", "-c", "usb-blaster", str(svf_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    shapes = [(128, 384), (320, 960), (384, 1152)]
    wls = []
    sdram_words: list[int] = []
    for idx, (dm, df) in enumerate(shapes):
        wl = make_ffn(dm, df, 4, seed=20261250 + idx)
        b1 = len(sdram_words); sdram_words.extend(wl["l1_words"])
        b2 = len(sdram_words); sdram_words.extend(wl["l2_words"])
        n_pkt = wl["l1_meta"]["n_tiles"] * wl["l1_meta"]["k_chunks"] + wl["l2_meta"]["n_tiles"] * wl["l2_meta"]["k_chunks"]
        wls.append({**wl, "shape": f"{dm}->{df}->{dm}", "l1_base": b1, "l2_base": b2, "n_pkt": n_pkt})

    targets = [
        ("e80_s80_p0", "top_sdram80.svf", 80.0),
        ("e90_s90_p0", "top_eng90_sd90.svf", 90.0),
    ]
    n_flashes = 10
    n_reps = 5
    all_rows = []

    try:
        for tag, svf_name, mhz in targets:
            svf_path = OF_DIR / svf_name
            ref_int = int(mhz) * 14
            base_preds = {
                w["shape"]: float(
                    predict_rtl_gen(w["l1_base"], w["l1_meta"], 4, mhz, mhz, ref_int, 8)
                    + predict_rtl_gen(w["l2_base"], w["l2_meta"], 4, mhz, mhz, ref_int, 8)
                )
                for w in wls
            }
            print(f"\n=== E1 Repeatability: {tag} ({n_flashes} independent SRAM re-flashes, n={n_reps}/shape) ===")
            for f_idx in range(1, n_flashes + 1):
                # Flash a different config (top.svf) in between on even flashes to force full PLL VCO/M/D change
                if f_idx > 1:
                    flash_svf(SVF_DEFAULT)
                flash_svf(svf_path)
                dev = UsbBlasterJtag()
                try:
                    dev.sdram_write_bulk_words(0, sdram_words)
                    flash_shifts = []
                    shape_res = {}
                    for w in wls:
                        dm, df = w["d_model"], w["d_ff"]
                        x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
                        hpost_base = len(x_words)
                        for i, xw in enumerate(x_words):
                            dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
                        dev.flush()
                        write_lut_256(dev, w["lut"], lut_bank=0)

                        cycs = []
                        for rep in range(n_reps):
                            c1 = run_sdram_stream_layer(
                                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                                4, dm, df, w["l1_meta"], w["l1_base"], 4, 1, 0, hpost_base, 1,
                            )
                            c2 = run_sdram_stream_layer(
                                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                                4, df, dm, w["l2_meta"], w["l2_base"], 4, 0, hpost_base, 0, 0,
                            )
                            if rep == 0:
                                y_exp = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]
                                assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"0-LSB fail {tag} f={f_idx}"
                            cycs.append(int(c1["total_cycles"] + c2["total_cycles"]))

                        meas_m = float(np.mean(cycs))
                        bp = base_preds[w["shape"]]
                        sh = (meas_m - bp) / float(w["n_pkt"])
                        flash_shifts.append(sh)
                        shape_res[w["shape"]] = {
                            "meas_mean": round(meas_m, 2),
                            "meas_sd": round(float(np.std(cycs, ddof=1)), 2),
                            "base_pred": round(bp, 2),
                            "shift_cyc_per_pkt": round(sh, 3),
                        }
                    row = {
                        "tag": tag,
                        "mhz": mhz,
                        "flash_idx": f_idx,
                        "mean_shift_cyc_per_pkt": round(float(np.mean(flash_shifts)), 3),
                        "shapes": shape_res,
                    }
                    all_rows.append(row)
                    print(
                        f"  [{tag} flash {f_idx:02d}/{n_flashes}] "
                        + " | ".join(f"{s}: {v['shift_cyc_per_pkt']:+.3f}" for s, v in shape_res.items())
                        + f" | mean={row['mean_shift_cyc_per_pkt']:+.3f} cyc/pkt"
                    )
                finally:
                    dev.close()
    finally:
        if SVF_DEFAULT.exists():
            flash_svf(SVF_DEFAULT)

    summary = {}
    for tag, _, _ in targets:
        sub = [r for r in all_rows if r["tag"] == tag]
        means = [r["mean_shift_cyc_per_pkt"] for r in sub]
        summary[tag] = {
            "n_flashes": len(sub),
            "flash_mean_shifts": means,
            "overall_mean": round(float(np.mean(means)), 3),
            "across_flash_sd": round(float(np.std(means, ddof=1)), 3),
            "min_flash_mean": round(float(min(means)), 3),
            "max_flash_mean": round(float(max(means)), 3),
        }

    OUT_JSON.write_text(json.dumps({"summary": summary, "flashes": all_rows}, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_JSON}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
