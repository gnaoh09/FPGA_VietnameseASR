#!/usr/bin/env python3
"""Round-6 verification analyses (Section 4 items 1, 2, 3):
  4.1: RTL simulation with clk1_phase_shift (0, 90, 180, 270 deg) at 80/80, 90/90, 40/40, 75/75 MHz.
  4.2: Adjusted R^2 and 95% CIs on sim-to-silicon residual regression (all 78, non-1:1 66, 1:1 12).
  4.3: Wall-clock runtime benchmark: analytical Python model vs Icarus Verilog RTL simulation.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_43_silicon_heldout import make_ffn, predict_rtl_gen

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
OUT_JSON = BUILD / "round6_analyses_results.json"


def parse_shape(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.replace("ffn_", "").split("_")]
    dm, df = parts[0], parts[1]
    return dm, df, max(math.ceil(dm / 256), math.ceil(df / 256))


def ols_summary(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    n, k = X.shape
    p = k - 1
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - p - 1)
    s2 = ss_res / max(1, n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(0.0, np.diag(cov)))
    tcrit = 1.993 if n > 60 else 2.365
    coeffs = {}
    for nm, b, s in zip(names, beta, se):
        coeffs[nm] = {
            "est": round(float(b), 4),
            "se": round(float(s), 4),
            "ci95": [round(float(b - tcrit * s), 4), round(float(b + tcrit * s), 4)],
        }
    return {"n": n, "p": p, "r2": round(r2, 4), "adj_r2": round(adj_r2, 4), "coeffs": coeffs}


def main() -> int:
    print("=== EXP 4.1: RTL SIM WITH CLK1_PHASE_SHIFT (0, 90, 180, 270 deg) ===")
    shapes = [(128, 384), (320, 960), (384, 1152)]
    wls = []
    addr = 0
    for idx, (dm, df) in enumerate(shapes):
        wl = make_ffn(dm, df, 4, seed=20261250 + idx)
        b1 = addr; addr += len(wl["l1_words"])
        b2 = addr; addr += len(wl["l2_words"])
        n_pkt = wl["l1_meta"]["n_tiles"] * wl["l1_meta"]["k_chunks"] + wl["l2_meta"]["n_tiles"] * wl["l2_meta"]["k_chunks"]
        wls.append({**wl, "shape": f"{dm}->{df}->{dm}", "l1_base": b1, "l2_base": b2, "n_pkt": n_pkt})

    sim_phase_rows = []
    for mhz in [80, 90, 40, 75]:
        vvp = BUILD / f"tb_78pt_0ppm_sd{mhz}.vvp"
        subprocess.run(
            [
                "iverilog", "-g2012", "-DSIMULATION", f"-Ptb_78pt_sim.CLK_MHZ_P={mhz}",
                "-o", str(vvp), "rtl/sdram_ctrl.v", "rtl/zipformer_engine.v", "tb/tb_78pt_sim.v",
            ],
            cwd=str(ROOT), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        half_ns = 500.0 / float(mhz)
        period_ns = 1000.0 / float(mhz)
        ref_int = mhz * 14
        for deg in ([0, 90, 180, 270] if mhz in [80, 40] else [0, 180]):
            ph_ns = (deg / 360.0) * period_ns
            shifts = []
            for w in wls:
                m1, m2 = w["l1_meta"], w["l2_meta"]
                base_pred = predict_rtl_gen(w["l1_base"], m1, 4, float(mhz), float(mhz), ref_int, 8) + \
                            predict_rtl_gen(w["l2_base"], m2, 4, float(mhz), float(mhz), ref_int, 8)
                cmd = [
                    "vvp", str(vvp),
                    f"+ENG_HALF_NS={half_ns:.6f}", f"+SDRAM_HALF_NS={half_ns:.6f}",
                    f"+SDRAM_PHASE_NS={ph_ns:.6f}",
                    f"+REF_INT={ref_int}", "+T_RFC=8",
                    f"+L1_KC={m1['k_chunk']}", f"+L1_NT={m1['n_tiles']}", f"+L1_KCH={m1['k_chunks']}",
                    f"+L1_WW={m1['w_words_per_chunk']}", f"+L1_BASE={w['l1_base']}",
                    f"+L2_KC={m2['k_chunk']}", f"+L2_NT={m2['n_tiles']}", f"+L2_KCH={m2['k_chunks']}",
                    f"+L2_WW={m2['w_words_per_chunk']}", f"+L2_BASE={w['l2_base']}",
                    f"+DM={w['d_model']}", f"+DF={w['d_ff']}",
                ]
                out = subprocess.check_output(cmd, text=True)
                m = re.search(r"\btot=(\d+)", out)
                assert m is not None, out
                sim_cyc = int(m.group(1))
                sh = (sim_cyc - base_pred) / float(w["n_pkt"])
                shifts.append((w["shape"], sim_cyc, round(base_pred, 1), round(sh, 3)))
                sim_phase_rows.append({
                    "mhz": mhz, "phase_deg": deg, "phase_ns": round(ph_ns, 4),
                    "shape": w["shape"], "n_pkt": w["n_pkt"],
                    "sim_cyc": sim_cyc, "base_pred": round(base_pred, 2),
                    "sim_shift_cyc_per_pkt": round(sh, 3),
                })
            print(
                f"  {mhz}/{mhz} MHz phase={deg:3d} deg (delay={ph_ns:6.3f} ns): "
                + " | ".join(f"{s}: sim={c} base={b} shift={sh:+.3f}" for s, c, b, sh in shifts)
            )

    print("\n=== EXP 4.2: ADJUSTED R2 & COEFF CIs FOR SIM-VS-SILICON REGRESSION ===")
    d78 = json.loads((BUILD / "three_way_78pt_0ppm_decomp.json").read_text())["rows"]
    reg_out = {}
    for label, sub in [
        ("all_78", d78),
        ("non_1to1_66", [r for r in d78 if not r["is_1to1"]]),
        ("only_1to1_12", [r for r in d78 if r["is_1to1"]]),
    ]:
        y = np.array([r["silicon_mean"] - r["sim_0ppm_tot"] for r in sub], dtype=float)
        M = np.array([r["m_tot"] for r in sub], dtype=float)
        dm = np.array([parse_shape(r["shape"])[0] for r in sub], dtype=float)
        km = np.array([parse_shape(r["shape"])[2] for r in sub], dtype=float)
        fs = np.array([r["sdram_mhz"] for r in sub], dtype=float)
        s_m = ols_summary(np.column_stack([np.ones_like(M), M]), y, ["b0", "M"])
        s_4 = ols_summary(np.column_stack([np.ones_like(M), M, km, dm, fs]), y, ["b0", "M", "Kmax", "d_m", "f_sd"])
        s_fs = ols_summary(np.column_stack([np.ones_like(M), dm, fs]), y, ["b0", "d_m", "f_sd"])
        reg_out[label] = {"1_M": s_m, "1_dm_fsd": s_fs, "1_M_Kmax_dm_fsd": s_4}
        print(f"  {label} [1, M]: R2={s_m['r2']:.4f}, adj_R2={s_m['adj_r2']:.4f}, coeffs={s_m['coeffs']}")
        print(f"  {label} [1, d_m, f_sd]: R2={s_fs['r2']:.4f}, adj_R2={s_fs['adj_r2']:.4f}, coeffs={s_fs['coeffs']}")
        print(f"  {label} [1, M, Kmax, d_m, f_sd]: R2={s_4['r2']:.4f}, adj_R2={s_4['adj_r2']:.4f}, coeffs={s_4['coeffs']}")

    print("\n=== EXP 4.3: MODEL SPEED VS RTL SIM SPEED (192->640->192, 80/90 MHz) ===")
    w640 = make_ffn(192, 640, 4, seed=42)
    m1, m2 = w640["l1_meta"], w640["l2_meta"]
    t0 = time.perf_counter()
    for _ in range(2000):
        _ = predict_rtl_gen(0, m1, 4, 80.0, 90.0, 1260, 8) + predict_rtl_gen(4160, m2, 4, 80.0, 90.0, 1260, 8)
    t_model_us = (time.perf_counter() - t0) / 2000.0 * 1e6

    vvp90 = BUILD / "tb_78pt_0ppm_sd90.vvp"
    cmd90 = [
        "vvp", str(vvp90), "+ENG_HALF_NS=6.250000", "+SDRAM_HALF_NS=5.555556",
        "+REF_INT=1260", "+T_RFC=8",
        f"+L1_KC={m1['k_chunk']}", f"+L1_NT={m1['n_tiles']}", f"+L1_KCH={m1['k_chunks']}",
        f"+L1_WW={m1['w_words_per_chunk']}", "+L1_BASE=0",
        f"+L2_KC={m2['k_chunk']}", f"+L2_NT={m2['n_tiles']}", f"+L2_KCH={m2['k_chunks']}",
        f"+L2_WW={m2['w_words_per_chunk']}", "+L2_BASE=4160",
        "+DM=192", "+DF=640",
    ]
    t0 = time.perf_counter()
    for _ in range(10):
        subprocess.check_output(cmd90, text=True)
    t_sim_ms = (time.perf_counter() - t0) / 10.0 * 1e3
    speedup = (t_sim_ms * 1e3) / t_model_us
    print(f"  Analytical Python model (1 FFN block = 2 layers, 64 packets): {t_model_us:.2f} us/eval")
    print(f"  Icarus Verilog vvp RTL sim (same 1 FFN block, ~65,300 cycles): {t_sim_ms:.2f} ms/eval ({speedup:.0f}x faster)")

    OUT_JSON.write_text(
        json.dumps(
            {
                "exp41_rtl_phase_sim": sim_phase_rows,
                "exp42_regression_adj_r2": reg_out,
                "exp43_speed_benchmark": {
                    "workload": "192->640->192 (T_tile=4, 2 layers, 64 packets, ~65,300 engine cycles)",
                    "analytical_python_us": round(t_model_us, 2),
                    "icarus_vvp_rtl_sim_ms": round(t_sim_ms, 2),
                    "speedup_factor": round(speedup, 1),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
