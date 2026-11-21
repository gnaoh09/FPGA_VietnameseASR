#!/usr/bin/env python3
"""Run Round-4 silicon experiments E7, E8, E9 on EP4CE10E22C8N:
  E7: 5.0-second long-window ratio of 64-bit free_clk_cnt vs sdram_cyc_cnt at 60/60, 80/80, 80/90, 80/100 MHz
      to directly verify exact rational (0 ppm relative) PLL clock ratios.
  E8: Frozen held-out 1:1 validation for B3 (+1.0 cyc/pkt) at 50/50, 70/70, 90/90 MHz on 3 unseen shapes
      (128->384->128, 320->960->320, 384->1152->384), n=10 repeats/pt, 0-LSB verified.
  E9: Rational 5:4 ratio (80/64 MHz) on the same 3 unseen shapes (n=10, 0-LSB verified).

Frozen predictions commit: 7aa1438ad9db3c821c09230ffe6703095ecb4942
Always restores build/output_files/top.svf in finally:.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpga_jtag_runner import OP_COMMIT_ACT, UsbBlasterJtag, pack_cfg1, pack_cfg2
from gen_golden_vectors import pack_words_128bit_i8
from run_43_silicon_heldout import make_ffn
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import run_bist_stats
from sdram_helpers import run_sdram_stream_layer, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = BUILD_DIR / "output_files"
SVF_DEFAULT = OF_DIR / "top.svf"
FROZEN_JSON = BUILD_DIR / "frozen_e8_e9_predictions.json"
OUT_JSON = BUILD_DIR / "e7_e9_silicon_results.json"


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def measure_long_window_ratio(dev: UsbBlasterJtag, window_s: float = 4.0) -> dict:
    """Measure Δfree_clk_cnt / Δsdram_cyc_cnt over a multi-second window so JTAG read-gap cancels out."""
    e0 = dev.read_resp(0x2004)
    s0 = dev.read_resp(0x200A)
    t0 = time.perf_counter()
    time.sleep(window_s)
    e1 = dev.read_resp(0x2004)
    s1 = dev.read_resp(0x200A)
    dt = time.perf_counter() - t0
    de = int(e1 - e0)
    ds = int(s1 - s0)
    eng_mhz = de / (dt * 1e6)
    sd_mhz = ds / (dt * 1e6)
    ratio = de / float(ds) if ds > 0 else 0.0
    return {
        "window_s": round(dt, 4),
        "d_eng_cyc": de,
        "d_sdram_cyc": ds,
        "eng_mhz": round(eng_mhz, 4),
        "sdram_mhz": round(sd_mhz, 4),
        "ratio_eng_over_sdram": round(ratio, 7),
    }


def main() -> int:
    frozen = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))["predictions"]
    shapes = [(128, 384), (320, 960), (384, 1152)]
    wls = []
    sdram_words: list[int] = []
    for idx, (dm, df) in enumerate(shapes):
        wl = make_ffn(dm, df, 4, seed=20261250 + idx)
        b1 = len(sdram_words); sdram_words.extend(wl["l1_words"])
        b2 = len(sdram_words); sdram_words.extend(wl["l2_words"])
        wls.append({**wl, "shape": f"{dm}->{df}->{dm}", "l1_base": b1, "l2_base": b2})

    res: dict = {"frozen_commit": "7aa1438ad9db3c821c09230ffe6703095ecb4942"}

    try:
        # E7: Long-window 64-bit ratio check on 60/60, 80/80, 80/90, 80/100
        e7_cases = [
            ("e60_s60", "top_eng60_sd60.svf", 60.0 / 60.0),
            ("e80_s80", "top_sdram80.svf",    80.0 / 80.0),
            ("e80_s90", "top_sdram90.svf",    80.0 / 90.0),
            ("e80_s100","top.svf",            80.0 / 100.0),
        ]
        e7_rows = []
        for tag, svf_name, nom_ratio in e7_cases:
            flash_svf(OF_DIR / svf_name)
            dev = UsbBlasterJtag()
            try:
                m = measure_long_window_ratio(dev, window_s=4.0)
                ppm_err = (m["ratio_eng_over_sdram"] - nom_ratio) / nom_ratio * 1e6
                row = {"tag": tag, "nom_ratio": round(nom_ratio, 7), **m, "rel_ppm_err": round(ppm_err, 2)}
                e7_rows.append(row)
                print(f"[E7 {tag}] ratio={m['ratio_eng_over_sdram']:.7f} vs nom={nom_ratio:.7f} ({ppm_err:+.2f} ppm)")
            finally:
                dev.close()
        res["E7_long_window_ppm"] = e7_rows

        # E8 & E9: Held-out 1:1 (50/50, 70/70, 90/90) + 5:4 (80/64)
        e8_e9_rows = []
        for tag in ("e50_s50", "e70_s70", "e90_s90", "e80_s64"):
            sub_preds = [p for p in frozen if p["tag"] == tag]
            svf_path = OF_DIR / sub_preds[0]["svf"]
            flash_svf(svf_path)
            dev = UsbBlasterJtag()
            try:
                mclk = measure_long_window_ratio(dev, window_s=2.0)
                if mclk["d_sdram_cyc"] == 0:
                    print(f"[{tag}] SDRAM clock did not lock (d_sdram_cyc=0); skipping.")
                    res.setdefault("skipped", []).append(tag)
                    continue
                bist = run_bist_stats(dev, 65536, mclk["sdram_mhz"])
                assert bist["err_cnt"] == 0, f"BIST failed on {tag}"
                dev.sdram_write_bulk_words(0, sdram_words)
                nom_ratio = sub_preds[0]["eng_mhz"] / float(sub_preds[0]["sdram_mhz"])
                ppm_err = (mclk["ratio_eng_over_sdram"] - nom_ratio) / nom_ratio * 1e6
                print(f"[{tag}] eng={mclk['eng_mhz']:.3f}M, sd={mclk['sdram_mhz']:.3f}M, ratio_ppm={ppm_err:+.2f}, BIST={bist['rd_MBps_1e6']} MB/s")

                for p, w in zip(sub_preds, wls):
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
                            4, w["d_model"], w["d_ff"], w["l1_meta"], w["l1_base"],
                            4, 1, 0, hpost_base, 1
                        )
                        c2 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            4, w["d_ff"], w["d_model"], w["l2_meta"], w["l2_base"],
                            4, 0, hpost_base, 0, 0
                        )
                        if rep == 0:
                            y_exp = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]
                            assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"0-LSB fail {tag} {w['shape']}"
                        tots.append(int(c1["total_cycles"] + c2["total_cycles"]))
                        stalls.append(int(c1["stall_cycles"] + c2["stall_cycles"]))
                    meas_m = float(np.mean(tots))
                    meas_sd = float(np.std(tots, ddof=1))
                    err_base = abs(p["pred_base_wb"] - meas_m) / meas_m * 100.0
                    err_ext = abs(p["pred_ext_b3"] - meas_m) / meas_m * 100.0
                    row = {
                        **p,
                        "ratio_ppm_err": round(ppm_err, 2),
                        "meas_mean": round(meas_m, 2),
                        "meas_sd": round(meas_sd, 2),
                        "err_base_wb_pct": round(err_base, 5),
                        "err_ext_b3_pct": round(err_ext, 5),
                        "base_minus_meas_cyc": round(p["pred_base_wb"] - meas_m, 2),
                        "ext_b3_minus_meas_cyc": round(p["pred_ext_b3"] - meas_m, 2),
                    }
                    e8_e9_rows.append(row)
                    print(
                        f"  [{w['shape']}] meas={meas_m:.1f}+/-{meas_sd:.2f} | "
                        f"Base={p['pred_base_wb']} ({err_base:.4f}%) | ExtB3={p['pred_ext_b3']} ({err_ext:.4f}%)"
                    )
            finally:
                dev.close()
        res["E8_E9_rows"] = e8_e9_rows
        OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved: {OUT_JSON}")
    finally:
        if SVF_DEFAULT.exists():
            flash_svf(SVF_DEFAULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
