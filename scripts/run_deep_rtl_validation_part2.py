#!/usr/bin/env python3
"""Deep RTL & Hardware Validation Part 2 on EP4CE10E22C8N:
  1. 60-point Out-of-Sample sweep: 6 real Zipformer-30M FFN shapes x 10 SDRAM frequencies
     {60, 70, 75, 80, 88, 90, 96, 100, 110, 120} MHz @ 80 MHz Engine, n=10 repeats/point
     with 0-LSB verification and BIST bandwidth at each frequency.
  2. CDC +3 (sdram) vs +4 (engine) separation: 6 shapes on (ENG=60, SDRAM=90) and
     (ENG=90, SDRAM=100), n=10 repeats/point with 0-LSB verification.
  3. Leave-One-Shape-Out CV (LOSO-CV) "grey-box" baseline fitting (c_page, c_cdc_sd, c_cdc_eng, t_rfc)
     vs pure zero-fit White-Box RTL vs BIST bandwidth baseline on the exact same 60-point set.

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
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks, run_bist_stats
from run_oos_stall_validation import make_synthetic_integer_ffn, predict_layer_rtl
from sdram_helpers import run_sdram_stream_layer, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OF_DIR = BUILD_DIR / "output_files"
SVF_DEFAULT = OF_DIR / "top.svf"
OUT_JSON = BUILD_DIR / "deep_rtl_validation_part2.json"

T_TILE = 4
RD_LAT = 4
N_REPEATS = 10

SHAPES_6 = [
    {"name": "ffn_192_384_192", "d_model": 192, "d_ff": 384},
    {"name": "ffn_192_512_192", "d_model": 192, "d_ff": 512},
    {"name": "ffn_192_640_192", "d_model": 192, "d_ff": 640},
    {"name": "ffn_256_576_256", "d_model": 256, "d_ff": 576},
    {"name": "ffn_256_768_256", "d_model": 256, "d_ff": 768},
    {"name": "ffn_256_960_256", "d_model": 256, "d_ff": 960},
]

BITSTREAMS = [
    {"tag": "e80_s60",  "eng_mhz": 80, "sdram_mhz": 60,  "svf": OF_DIR / "top_sdram60.svf",    "group": "oos_60pt"},
    {"tag": "e80_s70",  "eng_mhz": 80, "sdram_mhz": 70,  "svf": OF_DIR / "top_sdram70.svf",    "group": "oos_60pt"},
    {"tag": "e80_s75",  "eng_mhz": 80, "sdram_mhz": 75,  "svf": OF_DIR / "top_sdram75.svf",    "group": "oos_60pt"},
    {"tag": "e80_s80",  "eng_mhz": 80, "sdram_mhz": 80,  "svf": OF_DIR / "top_sdram80.svf",    "group": "oos_60pt"},
    {"tag": "e80_s88",  "eng_mhz": 80, "sdram_mhz": 88,  "svf": OF_DIR / "top_sdram88.svf",    "group": "oos_60pt"},
    {"tag": "e80_s90",  "eng_mhz": 80, "sdram_mhz": 90,  "svf": OF_DIR / "top_sdram90.svf",    "group": "oos_60pt"},
    {"tag": "e80_s96",  "eng_mhz": 80, "sdram_mhz": 96,  "svf": OF_DIR / "top_sdram96.svf",    "group": "oos_60pt"},
    {"tag": "e80_s100", "eng_mhz": 80, "sdram_mhz": 100, "svf": OF_DIR / "top.svf",            "group": "oos_60pt"},
    {"tag": "e80_s110", "eng_mhz": 80, "sdram_mhz": 110, "svf": OF_DIR / "top_sdram110.svf",   "group": "oos_60pt"},
    {"tag": "e80_s120", "eng_mhz": 80, "sdram_mhz": 120, "svf": OF_DIR / "top_sdram120.svf",   "group": "oos_60pt"},
    {"tag": "e60_s90",  "eng_mhz": 60, "sdram_mhz": 90,  "svf": OF_DIR / "top_eng60_sd90.svf", "group": "cdc_sep"},
    {"tag": "e60_s60",  "eng_mhz": 60, "sdram_mhz": 60,  "svf": OF_DIR / "top_eng60_sd60.svf", "group": "cdc_sep"},
    {"tag": "e70_s100", "eng_mhz": 70, "sdram_mhz": 100, "svf": OF_DIR / "top_eng70_sd100.svf","group": "cdc_sep"},
]


def flash_svf(svf_path: Path) -> None:
    print(f"[Flash] {svf_path.name}")
    subprocess.run(["openFPGALoader", "-c", "usb-blaster", str(svf_path)], check=True)


def predict_layer_param(
    base_w128: int,
    layer_meta: dict,
    eng_mhz: float,
    sdram_mhz: float,
    ref_interval: int,
    c_page: float = 14.0,
    c_cdc_sd: float = 3.0,
    c_cdc_eng: float = 4.0,
    t_rfc: float = 8.0,
) -> float:
    n_tiles = int(layer_meta["n_tiles"])
    k_chunks = int(layer_meta["k_chunks"])
    k_chunk = int(layer_meta["k_chunk"])
    pkt_words = int(layer_meta["pkt_words"])
    num_chunks = n_tiles * k_chunks

    c_tiles = []
    c_dmas = []
    addr = base_w128
    for _ in range(n_tiles):
        for kc in range(k_chunks):
            has_epi = (kc == k_chunks - 1)
            c_tile = 4 * k_chunk + (13 if has_epi else 6)
            pages = ((addr + pkt_words - 1) // 32) - (addr // 32) + 1
            sd_cyc = (8 * pkt_words + c_page * pages + 1) * (1.0 + t_rfc / float(ref_interval)) + c_cdc_sd
            c_dma = sd_cyc * (eng_mhz / sdram_mhz) + c_cdc_eng
            c_tiles.append(c_tile)
            c_dmas.append(c_dma)
            addr += pkt_words
    stall = c_dmas[0]
    for idx in range(1, num_chunks):
        stall += max(0.0, c_dmas[idx] - c_tiles[idx - 1])
    return float(sum(c_tiles)) + stall


def fit_ls_bounded(res_fn, x0: list[float], lo: list[float], hi: list[float], max_iter: int = 80):
    """Pure-NumPy damped Gauss-Newton least-squares with box bounds."""
    x = np.array(x0, dtype=np.float64)
    lo_a = np.array(lo, dtype=np.float64)
    hi_a = np.array(hi, dtype=np.float64)
    eps = 1e-4
    lam = 1e-3
    r = res_fn(x)
    cost = float(np.dot(r, r))
    for _ in range(max_iter):
        J = np.zeros((len(r), len(x)), dtype=np.float64)
        for j in range(len(x)):
            xp = x.copy()
            xp[j] = min(hi_a[j], x[j] + eps)
            xm = x.copy()
            xm[j] = max(lo_a[j], x[j] - eps)
            J[:, j] = (res_fn(xp) - res_fn(xm)) / max(xp[j] - xm[j], 1e-12)
        A = J.T @ J + lam * np.eye(len(x))
        g = J.T @ r
        step = np.linalg.solve(A, -g)
        x_new = np.clip(x + step, lo_a, hi_a)
        r_new = res_fn(x_new)
        cost_new = float(np.dot(r_new, r_new))
        if cost_new < cost:
            rel = abs(cost - cost_new) / max(cost, 1e-12)
            x, r, cost = x_new, r_new, cost_new
            lam = max(lam * 0.5, 1e-7)
            if rel < 1e-10 and np.linalg.norm(step) < 1e-6:
                break
        else:
            lam *= 4.0
    return x


def predict_bist_baseline(wl: dict, eng_mhz: float, bist_rd_mbps: float) -> float:
    total_bytes = wl["l1_meta"]["total_bytes"] + wl["l2_meta"]["total_bytes"]
    dma_eng_cyc = (total_bytes / (bist_rd_mbps * 1e6)) * (eng_mhz * 1e6)
    act1 = wl["l1_meta"]["n_tiles"] * sum(
        4 * wl["l1_meta"]["k_chunk"] + (13 if kc == wl["l1_meta"]["k_chunks"] - 1 else 6)
        for kc in range(wl["l1_meta"]["k_chunks"])
    )
    act2 = wl["l2_meta"]["n_tiles"] * sum(
        4 * wl["l2_meta"]["k_chunk"] + (13 if kc == wl["l2_meta"]["k_chunks"] - 1 else 6)
        for kc in range(wl["l2_meta"]["k_chunks"])
    )
    return float(max(act1 + act2, dma_eng_cyc))


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    workloads = []
    all_sdram_words: list[int] = []
    for idx, spec in enumerate(SHAPES_6):
        wl = make_synthetic_integer_ffn(spec["d_model"], spec["d_ff"], seed=20261130 + idx)
        l1_base = len(all_sdram_words)
        all_sdram_words.extend(wl["l1_words"])
        l2_base = len(all_sdram_words)
        all_sdram_words.extend(wl["l2_words"])
        workloads.append({**spec, **wl, "l1_base": l1_base, "l2_base": l2_base})

    print(f"Total SDRAM payload for 6 shapes: {len(all_sdram_words)} x 128b ({len(all_sdram_words)*16/1024:.1f} KiB)")

    records = []
    freq_meta = []

    try:
        for bs in BITSTREAMS:
            assert bs["svf"].exists(), f"Missing {bs['svf']}"
            flash_svf(bs["svf"])
            dev = UsbBlasterJtag()
            try:
                eng_meas, sd_meas, _, _ = measure_pll_clocks(dev)
                bist = run_bist_stats(dev, 65536, sd_meas)
                assert bist["err_cnt"] == 0, f"BIST failed on {bs['tag']}"
                dev.sdram_write_bulk_words(0, all_sdram_words)

                ref_int = bs["sdram_mhz"] * 14
                t_rfc = 9 if bs["sdram_mhz"] > 100 else 8
                freq_meta.append({
                    "tag": bs["tag"],
                    "group": bs["group"],
                    "nom_eng_mhz": bs["eng_mhz"],
                    "nom_sdram_mhz": bs["sdram_mhz"],
                    "meas_eng_mhz": eng_meas,
                    "meas_sdram_mhz": sd_meas,
                    "bist_rd_MBps": bist["rd_MBps_1e6"],
                    "bist_rd_pct": bist["rd_pct_peak"],
                })
                print(f"[{bs['tag']}] eng={eng_meas:.3f}M, sdram={sd_meas:.3f}M, BIST={bist['rd_MBps_1e6']} MB/s")

                for w in workloads:
                    x_words = [int(s, 16) for s in pack_words_128bit_i8(w["x_q"])]
                    hpost_base = len(x_words)
                    for i, xw in enumerate(x_words):
                        dev.queue_write_128b(OP_COMMIT_ACT, i, xw)
                    dev.flush()
                    write_lut_256(dev, w["lut"], lut_bank=0)

                    tot_runs, stall_runs = [], []
                    for rep in range(N_REPEATS):
                        c1 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            T_TILE, w["d_model"], w["d_ff"], w["l1_meta"], w["l1_base"],
                            RD_LAT, 1, 0, hpost_base, 1
                        )
                        c2 = run_sdram_stream_layer(
                            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                            T_TILE, w["d_ff"], w["d_model"], w["l2_meta"], w["l2_base"],
                            RD_LAT, 0, hpost_base, 0, 0
                        )
                        if rep == 0:
                            y_exp = [int(s, 16) for s in pack_words_128bit_i8(w["y_q"])]
                            assert all(dev.read_128b(0, i) == exp for i, exp in enumerate(y_exp)), f"0-LSB fail {bs['tag']} {w['name']}"
                        tot_runs.append(int(c1["total_cycles"] + c2["total_cycles"]))
                        stall_runs.append(int(c1["stall_cycles"] + c2["stall_cycles"]))

                    p1 = predict_layer_param(w["l1_base"], w["l1_meta"], bs["eng_mhz"], bs["sdram_mhz"], ref_int, 14.0, 3.0, 4.0, float(t_rfc))
                    p2 = predict_layer_param(w["l2_base"], w["l2_meta"], bs["eng_mhz"], bs["sdram_mhz"], ref_int, 14.0, 3.0, 4.0, float(t_rfc))
                    wb_pred = p1 + p2
                    bist_pred = predict_bist_baseline(w, float(bs["eng_mhz"]), bist["rd_MBps_1e6"])
                    meas_mean = float(np.mean(tot_runs))
                    meas_sd = float(np.std(tot_runs, ddof=1))
                    wb_err_pct = abs(wb_pred - meas_mean) / meas_mean * 100.0
                    bist_err_pct = abs(bist_pred - meas_mean) / meas_mean * 100.0
                    records.append({
                        "tag": bs["tag"],
                        "group": bs["group"],
                        "shape": w["name"],
                        "eng_mhz": bs["eng_mhz"],
                        "sdram_mhz": bs["sdram_mhz"],
                        "ref_interval": ref_int,
                        "t_rfc_nom": t_rfc,
                        "l1_base": w["l1_base"],
                        "l2_base": w["l2_base"],
                        "meas_tot_mean": round(meas_mean, 2),
                        "meas_tot_sd": round(meas_sd, 2),
                        "meas_stall_mean": round(float(np.mean(stall_runs)), 2),
                        "meas_stall_sd": round(float(np.std(stall_runs, ddof=1)), 2),
                        "wb_pred": round(wb_pred, 2),
                        "wb_err_pct": round(wb_err_pct, 5),
                        "bist_pred": round(bist_pred, 2),
                        "bist_err_pct": round(bist_err_pct, 5),
                    })
                    print(
                        f"  [{w['name']}] meas={meas_mean:.1f}+/-{meas_sd:.2f} | "
                        f"WB={wb_pred:.1f} ({wb_err_pct:.4f}%) | BIST={bist_pred:.1f} ({bist_err_pct:.3f}%)"
                    )
            finally:
                dev.close()
            OUT_JSON.write_text(json.dumps({"device": "EP4CE10E22C8N", "n_repeats": N_REPEATS, "freq_meta": freq_meta, "records": records}, indent=2), encoding="utf-8")

        # LOSO-CV Grey-box baseline on the 60-point OOS set
        oos_recs = [r for r in records if r["group"] == "oos_60pt"]
        wl_map = {w["name"]: w for w in workloads}
        grey_errs = []
        grey_folds = []
        for hold_shape in [s["name"] for s in SHAPES_6]:
            train = [r for r in oos_recs if r["shape"] != hold_shape]
            test = [r for r in oos_recs if r["shape"] == hold_shape]

            def res_fn(theta):
                c_page, c_cdc_sd, c_cdc_eng, t_rfc_fit = theta
                out = []
                for r in train:
                    w = wl_map[r["shape"]]
                    p1 = predict_layer_param(r["l1_base"], w["l1_meta"], r["eng_mhz"], r["sdram_mhz"], r["ref_interval"], c_page, c_cdc_sd, c_cdc_eng, t_rfc_fit)
                    p2 = predict_layer_param(r["l2_base"], w["l2_meta"], r["eng_mhz"], r["sdram_mhz"], r["ref_interval"], c_page, c_cdc_sd, c_cdc_eng, t_rfc_fit)
                    out.append(((p1 + p2) - r["meas_tot_mean"]) / r["meas_tot_mean"])
                return np.array(out)

            opt_x = fit_ls_bounded(res_fn, x0=[14.0, 3.0, 4.0, 8.0], lo=[0.0, 0.0, 0.0, 0.0], hi=[30.0, 20.0, 20.0, 20.0])
            cp, csd, ceng, trfc = [float(v) for v in opt_x]
            fold_errs = []
            for r in test:
                w = wl_map[r["shape"]]
                p1 = predict_layer_param(r["l1_base"], w["l1_meta"], r["eng_mhz"], r["sdram_mhz"], r["ref_interval"], cp, csd, ceng, trfc)
                p2 = predict_layer_param(r["l2_base"], w["l2_meta"], r["eng_mhz"], r["sdram_mhz"], r["ref_interval"], cp, csd, ceng, trfc)
                e = abs((p1 + p2) - r["meas_tot_mean"]) / r["meas_tot_mean"] * 100.0
                fold_errs.append(e)
                grey_errs.append(e)
            grey_folds.append({
                "held_out_shape": hold_shape,
                "fitted_c_page": round(cp, 3),
                "fitted_c_cdc_sd": round(csd, 3),
                "fitted_c_cdc_eng": round(ceng, 3),
                "fitted_t_rfc": round(trfc, 3),
                "test_mean_err_pct": round(float(np.mean(fold_errs)), 5),
                "test_max_err_pct": round(float(np.max(fold_errs)), 5),
            })

        # Joint fit on all 72 points (60 OOS + 12 CDC) to separate c_cdc_sd (+3) and c_cdc_eng (+4)
        def res_cdc(theta):
            c_cdc_sd, c_cdc_eng = theta
            out = []
            for r in records:
                w = wl_map[r["shape"]]
                p1 = predict_layer_param(r["l1_base"], w["l1_meta"], r["eng_mhz"], r["sdram_mhz"], r["ref_interval"], 14.0, c_cdc_sd, c_cdc_eng, float(r["t_rfc_nom"]))
                p2 = predict_layer_param(r["l2_base"], w["l2_meta"], r["eng_mhz"], r["sdram_mhz"], r["ref_interval"], 14.0, c_cdc_sd, c_cdc_eng, float(r["t_rfc_nom"]))
                out.append((p1 + p2) - r["meas_tot_mean"])
            return np.array(out)

        opt_cdc_x = fit_ls_bounded(res_cdc, x0=[3.0, 4.0], lo=[0.0, 0.0], hi=[15.0, 15.0])
        wb_errs = [r["wb_err_pct"] for r in oos_recs]
        bist_errs = [r["bist_err_pct"] for r in oos_recs]

        summary = {
            "oos_60pt": {
                "n_points": len(oos_recs),
                "wb_mean_pct": round(float(np.mean(wb_errs)), 5),
                "wb_median_pct": round(float(np.median(wb_errs)), 5),
                "wb_p95_pct": round(float(np.percentile(wb_errs, 95)), 5),
                "wb_max_pct": round(float(np.max(wb_errs)), 5),
                "grey_loso_mean_pct": round(float(np.mean(grey_errs)), 5),
                "grey_loso_median_pct": round(float(np.median(grey_errs)), 5),
                "grey_loso_p95_pct": round(float(np.percentile(grey_errs, 95)), 5),
                "grey_loso_max_pct": round(float(np.max(grey_errs)), 5),
                "bist_mean_pct": round(float(np.mean(bist_errs)), 5),
                "bist_median_pct": round(float(np.median(bist_errs)), 5),
                "bist_p95_pct": round(float(np.percentile(bist_errs, 95)), 5),
                "bist_max_pct": round(float(np.max(bist_errs)), 5),
            },
            "cdc_separation_72pt": {
                "fitted_c_cdc_sd_rtl3": round(float(opt_cdc_x[0]), 3),
                "fitted_c_cdc_eng_rtl4": round(float(opt_cdc_x[1]), 3),
            },
            "grey_loso_folds": grey_folds,
        }
        print("\n=== SUMMARY 60-PT OOS + 12-PT CDC ===")
        print(json.dumps(summary, indent=2))

        payload = {
            "device": "EP4CE10E22C8N",
            "n_repeats": N_REPEATS,
            "freq_meta": freq_meta,
            "records": records,
            "summary": summary,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved: {OUT_JSON}")
    finally:
        if SVF_DEFAULT.exists():
            flash_svf(SVF_DEFAULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
