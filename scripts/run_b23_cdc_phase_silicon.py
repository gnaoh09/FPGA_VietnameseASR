#!/usr/bin/env python3
"""B2.3 Controlled 1:1 CDC & PLL Phase-Alignment Silicon Experiment.

Tests:
  1. Controlled PLL c1 phase-shift sweep at 80/80 MHz: {0, 90, 180, 270} deg
     and at 90/90 MHz: {0, 180} deg.
  2. Additional 1:1 clock frequencies: 40/40 MHz and 75/75 MHz (0 deg).
  3. Compares silicon cycles (3 unseen shapes, n=10 repeats each, 0-LSB verified)
     against Base White-Box and +1.0 cyc/pkt ExtB3 predictions, and extracts
     the empirical per-packet CDC shift (meas - base) / n_packets.

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
from run_e7_e9_silicon import measure_long_window_ratio
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import run_bist_stats
from sdram_helpers import run_sdram_stream_layer, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = BUILD_DIR / "output_files"
SVF_DEFAULT = OF_DIR / "top.svf"
OUT_JSON = BUILD_DIR / "b23_cdc_phase_silicon_results.json"


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def main() -> int:
    shapes = [(128, 384), (320, 960), (384, 1152)]
    wls = []
    sdram_words: list[int] = []
    for idx, (dm, df) in enumerate(shapes):
        wl = make_ffn(dm, df, 4, seed=20261250 + idx)
        b1 = len(sdram_words); sdram_words.extend(wl["l1_words"])
        b2 = len(sdram_words); sdram_words.extend(wl["l2_words"])
        wls.append({**wl, "shape": f"{dm}->{df}->{dm}", "l1_base": b1, "l2_base": b2})

    configs = [
        ("e80_s80_p0",   "top_sdram80.svf",          80.0, 80.0, 0),
        ("e80_s80_p90",  "top_eng80_sd80_p90.svf",   80.0, 80.0, 90),
        ("e80_s80_p180", "top_eng80_sd80_p180.svf",  80.0, 80.0, 180),
        ("e80_s80_p270", "top_eng80_sd80_p270.svf",  80.0, 80.0, 270),
        ("e90_s90_p0",   "top_eng90_sd90.svf",       90.0, 90.0, 0),
        ("e90_s90_p180", "top_eng90_sd90_p180.svf",  90.0, 90.0, 180),
        ("e40_s40_p0",   "top_eng40_sd40.svf",       40.0, 40.0, 0),
        ("e75_s75_p0",   "top_eng75_sd75.svf",       75.0, 75.0, 0),
    ]

    rows = []
    try:
        for tag, svf_name, f_eng, f_sd, phase_deg in configs:
            svf_path = OF_DIR / svf_name
            if not svf_path.exists():
                print(f"[SKIP] Missing {svf_path}")
                rows.append({"tag": tag, "status": "missing_bitstream"})
                continue

            flash_svf(svf_path)
            dev = UsbBlasterJtag()
            try:
                clk_m = measure_long_window_ratio(dev, window_s=2.0)
                if clk_m["d_sdram_cyc"] == 0:
                    print(f"[SKIP {tag}] SDRAM PLL c1 did not lock (d_sdram_cyc=0)")
                    rows.append({"tag": tag, "eng_mhz": f_eng, "sdram_mhz": f_sd, "phase_deg": phase_deg, "status": "pll_c1_not_locked", **clk_m})
                    continue

                ppm = (clk_m["ratio_eng_over_sdram"] - 1.0) * 1e6
                bist = run_bist_stats(dev, 65536, clk_m["sdram_mhz"])
                if bist["err_cnt"] != 0:
                    print(f"[FAIL {tag}] BIST errors={bist['err_cnt']}")
                    rows.append({"tag": tag, "eng_mhz": f_eng, "sdram_mhz": f_sd, "phase_deg": phase_deg, "status": "bist_error", "err_cnt": bist["err_cnt"], **clk_m})
                    continue

                dev.sdram_write_bulk_words(0, sdram_words)
                print(f"[{tag}] eng={clk_m['eng_mhz']:.3f}M, sd={clk_m['sdram_mhz']:.3f}M, ratio_ppm={ppm:+.2f}, BIST={bist['rd_MBps_1e6']} MB/s")

                ref_int = int(f_sd) * 14
                t_rfc = 9 if f_sd > 100 else 8
                for w in wls:
                    dm, df = w["d_model"], w["d_ff"]
                    n_pkt = w["l1_meta"]["n_tiles"] * w["l1_meta"]["k_chunks"] + w["l2_meta"]["n_tiles"] * w["l2_meta"]["k_chunks"]
                    p1 = predict_rtl_gen(w["l1_base"], w["l1_meta"], 4, f_eng, f_sd, ref_int, t_rfc)
                    p2 = predict_rtl_gen(w["l2_base"], w["l2_meta"], 4, f_eng, f_sd, ref_int, t_rfc)
                    base_pred = float(p1 + p2)
                    ext_pred = base_pred + float(n_pkt)

                    x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
                    hpost_base = len(x_words)
                    for i, xw in enumerate(x_words):
                        dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
                    dev.flush()
                    write_lut_256(dev, w["lut"], lut_bank=0)

                    cycs = []
                    for rep in range(10):
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
                            assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"0-LSB fail {tag} {w['shape']}"
                        cycs.append(int(c1["total_cycles"] + c2["total_cycles"]))

                    meas_mean = float(np.mean(cycs))
                    meas_sd = float(np.std(cycs, ddof=1))
                    base_err_pct = abs(base_pred - meas_mean) / meas_mean * 100.0
                    ext_err_pct = abs(ext_pred - meas_mean) / meas_mean * 100.0
                    shift_per_pkt = (meas_mean - base_pred) / float(n_pkt)
                    row = {
                        "tag": tag,
                        "status": "ok",
                        "eng_mhz": f_eng,
                        "sdram_mhz": f_sd,
                        "phase_deg": phase_deg,
                        "rel_ppm_err": round(ppm, 2),
                        "shape": w["shape"],
                        "n_packets": n_pkt,
                        "meas_mean": round(meas_mean, 2),
                        "meas_sd": round(meas_sd, 2),
                        "base_pred": round(base_pred, 2),
                        "ext_b3_pred": round(ext_pred, 2),
                        "base_err_pct": round(base_err_pct, 4),
                        "ext_b3_err_pct": round(ext_err_pct, 4),
                        "emp_shift_cyc_per_pkt": round(shift_per_pkt, 3),
                    }
                    rows.append(row)
                    print(
                        f"  [{tag} {w['shape']:13s}] meas={meas_mean:8.1f}±{meas_sd:4.1f} "
                        f"base={base_pred:8.1f} ({base_err_pct:.4f}%) "
                        f"extB3={ext_pred:8.1f} ({ext_err_pct:.4f}%) "
                        f"shift={shift_per_pkt:+.3f} cyc/pkt"
                    )
            finally:
                dev.close()
    finally:
        if SVF_DEFAULT.exists():
            flash_svf(SVF_DEFAULT)

    OUT_JSON.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
