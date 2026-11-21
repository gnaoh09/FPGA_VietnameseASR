#!/usr/bin/env python3
"""Reproducible Major-Revision analyses for Reviewer Reports #1-#3:
1. All 3 white-box variants (Base Duty-Cycle, Stateful Drop-in, RTL-Exact S_IDLE-Overlap)
   + Grey-Box_60 evaluated on the exact same 45-condition non-1:1 held-out set,
   each of its 6 suites (C1, C2, E9, B1_T4, B1_T8, Val43), the 18 frozen non-1:1 points,
   and 60-cond / 48-non-pilot / 54-non-1:1 / all-78 sets.
2. Grey-Box LOSO-CV Stall MAE, Stall MAPE, and total-cycle error paired on 60 / 48 / 54 sets,
   plus out-of-range extrapolation to T_tile=8 (B1_T8 and 80/40 MHz core).
3. Stateful recurrence Jensen-gap anatomy, initial-phase sensitivity, offset 1 vs 15 contrast,
   32-offset sweeps (192->640 and 192->512), and runtime benchmark.
4. Raw 10-repeat lag-1 autocorrelation and effective SE of the mean.
5. Stronger 60-condition baselines, upgraded clock selection, and padded 3-utterance Zipformer check.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
sys.path.insert(0, str(ROOT / "scripts"))

from run_deep_rtl_validation_part2 import SHAPES_6, predict_layer_param
from run_oos_stall_validation import make_synthetic_integer_ffn
from run_43_silicon_heldout import make_ffn, predict_rtl_gen


def page_word_counts(addr: int, P: int) -> list[int]:
    out: list[int] = []
    rem = P
    cur = addr
    while rem > 0:
        take = min(rem, 32 - (cur % 32))
        out.append(take)
        cur += take
        rem -= take
    return out


def compute_cycles(meta: dict, t_tile: int = 4) -> float:
    nt, kch, kc = meta["n_tiles"], meta["k_chunks"], meta["k_chunk"]
    return sum(
        t_tile * kc + (6.0 if (i % kch) < kch - 1 else 13.0)
        for i in range(nt * kch)
    )


def parse_sh(s: str) -> tuple[int, int]:
    s = s.replace("ffn_", "").replace("_", "->")
    p = [int(x) for x in s.split("->")]
    return p[0], p[1]


def stateful_rtl_model(
    base_addr: int,
    meta: dict,
    f_eng: float,
    f_sdram: float,
    ref_interval: int,
    t_rfc: int,
    init_ref_phase: float,
    req_cdc_sd: float = 3.0,
    ack_cdc_eng: float = 4.0,
    rearm_gap_eng: float = 1.0,
    t_tile: int = 4,
    rd_lat: int = 4,
    include_refresh: bool = True,
    cdc_absorbs_ref: bool = True,
) -> tuple[float, float, float]:
    n_tiles, k_chunks, k_chunk = meta["n_tiles"], meta["k_chunks"], meta["k_chunk"]
    P = 8 + meta["w_words_per_chunk"]
    M = n_tiles * k_chunks
    page_pen = rd_lat + 10
    ratio_e2s = f_sdram / f_eng
    ratio_s2e = f_eng / f_sdram

    ref_timer = float(init_ref_phase % ref_interval)
    ref_due = False

    def adv(dt: float) -> None:
        nonlocal ref_timer, ref_due
        if not include_refresh or dt <= 0.0:
            return
        nxt = ref_timer + dt
        if nxt >= ref_interval:
            ref_due = True
            ref_timer = nxt % ref_interval
        else:
            ref_timer = nxt

    def adv_idle(dt: float) -> float:
        nonlocal ref_timer, ref_due
        if not include_refresh or dt <= 0.0:
            return 0.0
        rem = dt
        if ref_due:
            svc = float(t_rfc) + 1.0
            if rem >= svc:
                ref_due = False
                adv(svc)
                rem -= svc
            else:
                adv(rem)
                ref_due = False
                ex = svc - rem
                adv(ex)
                return ex
        while rem > 0.0:
            to_fire = ref_interval - ref_timer
            if rem < to_fire:
                ref_timer += rem
                return 0.0
            rem -= to_fire
            ref_timer = 0.0
            svc = float(t_rfc) + 1.0
            if rem >= svc:
                adv(svc)
                rem -= svc
            else:
                adv(rem)
                ex = svc - rem
                adv(ex)
                return ex
        return 0.0

    t_sdone_eng = 0.0
    t_dma_end = [0.0] * M
    t_comp_end = [0.0] * M
    total_stall = 0.0
    total_compute = 0.0

    for i in range(M):
        kc = i % k_chunks
        addr = base_addr + i * P
        c_tile = t_tile * k_chunk + (6.0 if kc < k_chunks - 1 else 13.0)
        total_compute += c_tile

        if i == 0:
            t_req_eng = 0.0
            adv(req_cdc_sd)
            sd_cyc = req_cdc_sd
        else:
            t_req_eng = (
                (t_dma_end[i - 1] + rearm_gap_eng)
                if i == 1
                else max(t_dma_end[i - 1] + rearm_gap_eng, t_comp_end[i - 2])
            )
            if cdc_absorbs_ref:
                dt_idle_sd = max(0.0, (t_req_eng + req_cdc_sd * ratio_s2e - t_sdone_eng) * ratio_e2s)
                ex = adv_idle(dt_idle_sd)
            else:
                dt_comp_idle_sd = max(0.0, (t_req_eng - (t_dma_end[i - 1] + rearm_gap_eng)) * ratio_e2s)
                if dt_comp_idle_sd > 0.0:
                    ex = adv_idle(dt_comp_idle_sd)
                else:
                    ex = (float(t_rfc) + 1.0) if (include_refresh and ref_due) else 0.0
                    if include_refresh and ref_due:
                        ref_due = False
                        adv(float(t_rfc) + 1.0)
                adv(req_cdc_sd + (ack_cdc_eng + rearm_gap_eng) * ratio_e2s)
            sd_cyc = req_cdc_sd + ex

        pages = page_word_counts(addr, P)
        for j, b in enumerate(pages):
            p_cyc = 8.0 * b + page_pen
            adv(p_cyc)
            sd_cyc += p_cyc
            if j < len(pages) - 1 and include_refresh and ref_due:
                ref_due = False
                adv(float(t_rfc))
                sd_cyc += float(t_rfc)
        adv(1.0)
        sd_cyc += 1.0

        t_sdone_eng = t_req_eng + sd_cyc * ratio_s2e
        t_dma_end[i] = t_sdone_eng + ack_cdc_eng
        if i == 0:
            total_stall += t_dma_end[0]
            t_comp_start = t_dma_end[0]
        else:
            if t_dma_end[i] > t_comp_end[i - 1]:
                total_stall += t_dma_end[i] - t_comp_end[i - 1]
                t_comp_start = t_dma_end[i]
            else:
                t_comp_start = t_comp_end[i - 1]
        t_comp_end[i] = t_comp_start + c_tile

    return t_comp_end[-1], total_stall, total_compute


def avg_sf(
    base_addr: int,
    meta: dict,
    f_eng: float,
    f_sdram: float,
    ri: int,
    tr: int,
    n_ph: int = 32,
    **kw,
) -> tuple[float, float, float]:
    vals = [
        stateful_rtl_model(base_addr, meta, f_eng, f_sdram, ri, tr, (k + 0.5) * ri / n_ph, **kw)
        for k in range(n_ph)
    ]
    return float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])), vals[0][2]


def summarize_suite(items: list[tuple[float, float, float, float]]) -> dict:
    tot_p = [abs(p - m) / m * 100.0 for m, p, c, s in items]
    st_a = [abs(p - m) for m, p, c, s in items]
    st_m = [abs(p - m) / (m - c) * 100.0 for m, p, c, s in items]
    bias = [p - m for m, p, c, s in items]
    sds = [s for m, p, c, s in items]
    return {
        "n": len(items),
        "stall_mae_cyc": round(float(np.mean(st_a)), 2),
        "stall_med_cyc": round(float(np.median(st_a)), 2),
        "stall_max_cyc": round(float(np.max(st_a)), 2),
        "stall_mape_pct": round(float(np.mean(st_m)), 4),
        "stall_mape_max_pct": round(float(np.max(st_m)), 4),
        "signed_bias_cyc": round(float(np.mean(bias)), 2),
        "tot_err_mean_pct": round(float(np.mean(tot_p)), 5),
        "tot_err_max_pct": round(float(np.max(tot_p)), 5),
        "median_repeat_sd_cyc": round(float(np.median(sds)), 2),
    }


def layer_baselines(base_addr: int, meta: dict, f_eng: float, f_sdram: float, ref_int: int, t_rfc: int) -> dict:
    n_tiles, k_chunks, k_chunk = meta["n_tiles"], meta["k_chunks"], meta["k_chunk"]
    P = 8 + meta["w_words_per_chunk"]
    M = n_tiles * k_chunks
    ref_fac = 1.0 + t_rfc / ref_int
    c_tiles = [4 * k_chunk + (6.0 if (i % k_chunks) < k_chunks - 1 else 13.0) for i in range(M)]
    comp = sum(c_tiles)
    pages = [((base_addr + i * P + P - 1) // 32 - (base_addr + i * P) // 32 + 1) for i in range(M)]
    avg_p = sum(pages) / M
    cdmas = [((8 * P + 14 * p + 1) * ref_fac + 3.0) * (f_eng / f_sdram) + 4.0 for p in pages]
    wb_st = cdmas[0] + sum(max(0.0, cdmas[i] - c_tiles[i - 1]) for i in range(1, M))
    cdma_avg = ((8 * P + 14 * avg_p + 1) * ref_fac + 3.0) * (f_eng / f_sdram) + 4.0
    ap_st = cdma_avg + sum(max(0.0, cdma_avg - c_tiles[i - 1]) for i in range(1, M))
    comp_cdma0_st = cdmas[0]
    tot_dma_ideal = sum(((8 * P + 14 * p) * ref_fac) * (f_eng / f_sdram) for p in pages)
    ideal_tot = max(comp, tot_dma_ideal)
    return {
        "comp": comp,
        "wb": comp + wb_st,
        "avg_page": comp + ap_st,
        "comp_plus_cdma0": comp + comp_cdma0_st,
        "ideal_roofline": ideal_tot,
    }


def main() -> int:
    d78 = json.loads((BUILD / "three_way_78pt_0ppm_decomp.json").read_text())["rows"]
    d2_full = json.loads((BUILD / "deep_rtl_validation_part2.json").read_text())
    d2 = d2_full["records"]
    folds = {f["held_out_shape"]: f for f in d2_full["summary"]["grey_loso_folds"]}
    rec_map = {(r["tag"], r["shape"]): r for r in d2}
    wl_map = {
        s["name"]: make_synthetic_integer_ffn(s["d_model"], s["d_ff"], seed=20261130 + i)
        for i, s in enumerate(SHAPES_6)
    }

    # -------------------------------------------------------------------------
    # 1. Stateful Refresh vs Duty-Cycle across 78/60/48/54/66 + paired Grey-Box LOSO-CV
    # -------------------------------------------------------------------------
    pilot_shapes = {"ffn_192_384_192", "ffn_256_576_256", "ffn_256_960_256"}
    pilot_freqs = {60, 90, 100, 120}

    subsets_def = {
        "all_78": lambda r: True,
        "oos_60": lambda r: r["group"] == "oos_60pt",
        "non_pilot_48": lambda r: (
            r["group"] == "oos_60pt"
            and not (r["shape"] in pilot_shapes and r["sdram_mhz"] in pilot_freqs)
        ),
        "non_1to1_54": lambda r: r["group"] == "oos_60pt" and not r["is_1to1"],
        "non_1to1_66": lambda r: not r["is_1to1"],
    }

    stateful_subsets = {}
    for sname, cond in subsets_def.items():
        sub = [r for r in d78 if cond(r)]
        out_m = {}
        for mname, kw in [
            ("duty_cycle", None),
            ("stateful_dropin", dict(req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=0.0, cdc_absorbs_ref=False)),
            ("stateful_rtl_exact", dict(req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True)),
        ]:
            sim_pct, sil_pct, sim_s, sil_s, st_abs, st_mape = [], [], [], [], [], []
            for r in sub:
                rec = rec_map[(r["tag"], r["shape"])]
                w = wl_map[r["shape"]]
                eng, sd, ri, tr = float(r["eng_mhz"]), float(r["sdram_mhz"]), int(rec["ref_interval"]), int(rec["t_rfc_nom"])
                comp = compute_cycles(w["l1_meta"], 4) + compute_cycles(w["l2_meta"], 4)
                if kw is None:
                    pred = float(r["wb_pred"])
                else:
                    t1, _, _ = avg_sf(rec["l1_base"], w["l1_meta"], eng, sd, ri, tr, n_ph=32, **kw)
                    t2, _, _ = avg_sf(rec["l2_base"], w["l2_meta"], eng, sd, ri, tr, n_ph=32, **kw)
                    pred = t1 + t2
                sim_t = float(r["sim_0ppm_tot"])
                sil_t = float(r["silicon_mean"])
                sil_st = sil_t - comp
                sim_pct.append(abs(pred - sim_t) / sim_t * 100.0)
                sil_pct.append(abs(pred - sil_t) / sil_t * 100.0)
                sim_s.append(pred - sim_t)
                sil_s.append(pred - sil_t)
                st_abs.append(abs(pred - sil_t))
                st_mape.append(abs(pred - sil_t) / sil_st * 100.0)
            out_m[mname] = {
                "n": len(sub),
                "vs_sim_mean_pct": round(float(np.mean(sim_pct)), 4),
                "vs_sim_signed_cyc": round(float(np.mean(sim_s)), 2),
                "vs_sil_mean_pct": round(float(np.mean(sil_pct)), 4),
                "vs_sil_max_pct": round(float(np.max(sil_pct)), 4),
                "vs_sil_signed_cyc": round(float(np.mean(sil_s)), 2),
                "stall_mae_cyc": round(float(np.mean(st_abs)), 2),
                "stall_med_cyc": round(float(np.median(st_abs)), 2),
                "stall_mape_pct": round(float(np.mean(st_mape)), 4),
                "stall_mape_max_pct": round(float(np.max(st_mape)), 4),
            }
        # Also evaluate Grey-Box LOSO-CV on 60 / 48 / 54
        if sname in ("oos_60", "non_pilot_48", "non_1to1_54"):
            gb_abs, gb_mape, gb_tot, gb_s = [], [], [], []
            for r in sub:
                rec = rec_map[(r["tag"], r["shape"])]
                w = wl_map[r["shape"]]
                fp = folds[r["shape"]]
                eng, sd, ri = float(r["eng_mhz"]), float(r["sdram_mhz"]), int(rec["ref_interval"])
                comp = compute_cycles(w["l1_meta"], 4) + compute_cycles(w["l2_meta"], 4)
                pred = (
                    predict_layer_param(rec["l1_base"], w["l1_meta"], eng, sd, ri, fp["fitted_c_page"], fp["fitted_c_cdc_sd"], fp["fitted_c_cdc_eng"], fp["fitted_t_rfc"])
                    + predict_layer_param(rec["l2_base"], w["l2_meta"], eng, sd, ri, fp["fitted_c_page"], fp["fitted_c_cdc_sd"], fp["fitted_c_cdc_eng"], fp["fitted_t_rfc"])
                )
                sil_t = float(r["silicon_mean"])
                gb_abs.append(abs(pred - sil_t))
                gb_mape.append(abs(pred - sil_t) / (sil_t - comp) * 100.0)
                gb_tot.append(abs(pred - sil_t) / sil_t * 100.0)
                gb_s.append(pred - sil_t)
            out_m["grey_box_losocv"] = {
                "n": len(sub),
                "vs_sil_mean_pct": round(float(np.mean(gb_tot)), 4),
                "vs_sil_max_pct": round(float(np.max(gb_tot)), 4),
                "vs_sil_signed_cyc": round(float(np.mean(gb_s)), 2),
                "stall_mae_cyc": round(float(np.mean(gb_abs)), 2),
                "stall_med_cyc": round(float(np.median(gb_abs)), 2),
                "stall_mape_pct": round(float(np.mean(gb_mape)), 4),
                "stall_mape_max_pct": round(float(np.max(gb_mape)), 4),
            }
        stateful_subsets[sname] = out_m

    # Per-frequency stateful vs base on 60-condition campaign (for Fig 5a)
    oos_rows = [r for r in d78 if r["group"] == "oos_60pt"]
    per_freq_60 = []
    for sd_f in sorted(set(int(r["sdram_mhz"]) for r in oos_rows)):
        frows = [r for r in oos_rows if int(r["sdram_mhz"]) == sd_f]
        bist_a, wb_a, sf_di_a, sf_rtl_a, sim_a = [], [], [], [], []
        wb_m, sf_di_m = [], []
        for r in frows:
            rec = rec_map[(r["tag"], r["shape"])]
            w = wl_map[r["shape"]]
            eng, sd, ri, tr = float(r["eng_mhz"]), float(r["sdram_mhz"]), int(rec["ref_interval"]), int(rec["t_rfc_nom"])
            comp = compute_cycles(w["l1_meta"], 4) + compute_cycles(w["l2_meta"], 4)
            sil_t = float(r["silicon_mean"])
            sil_st = sil_t - comp
            sf_di = (
                avg_sf(rec["l1_base"], w["l1_meta"], eng, sd, ri, tr, n_ph=32, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=0.0, cdc_absorbs_ref=False)[0]
                + avg_sf(rec["l2_base"], w["l2_meta"], eng, sd, ri, tr, n_ph=32, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=0.0, cdc_absorbs_ref=False)[0]
            )
            sf_rtl = (
                avg_sf(rec["l1_base"], w["l1_meta"], eng, sd, ri, tr, n_ph=32, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True)[0]
                + avg_sf(rec["l2_base"], w["l2_meta"], eng, sd, ri, tr, n_ph=32, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True)[0]
            )
            bist_a.append(abs(float(r["bist_pred"]) - sil_t))
            wb_a.append(abs(float(r["wb_pred"]) - sil_t))
            sf_di_a.append(abs(sf_di - sil_t))
            sf_rtl_a.append(abs(sf_rtl - sil_t))
            sim_a.append(abs(float(r["sim_0ppm_tot"]) - sil_t))
            wb_m.append(abs(float(r["wb_pred"]) - sil_t) / sil_st * 100.0)
            sf_di_m.append(abs(sf_di - sil_t) / sil_st * 100.0)
        per_freq_60.append({
            "sdram_mhz": sd_f,
            "bist_stall_mae_cyc": round(float(np.mean(bist_a)), 2),
            "wb_stall_mae_cyc": round(float(np.mean(wb_a)), 2),
            "sf_dropin_stall_mae_cyc": round(float(np.mean(sf_di_a)), 2),
            "sf_rtl_stall_mae_cyc": round(float(np.mean(sf_rtl_a)), 2),
            "sim_0ppm_stall_mae_cyc": round(float(np.mean(sim_a)), 2),
            "wb_stall_mape_pct": round(float(np.mean(wb_m)), 4),
            "sf_dropin_stall_mape_pct": round(float(np.mean(sf_di_m)), 4),
        })

    # Offset 1 vs 15 contrast + phase sensitivity + Jensen anatomy
    w640 = make_synthetic_integer_ffn(192, 640, seed=20261121)
    offset_contrast = {}
    for inc_ref, lbl in [(True, "refresh_on"), (False, "refresh_off")]:
        o1_l1 = avg_sf(16385, w640["l1_meta"], 80.0, 90.0, 1260, 8, n_ph=64, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True, include_refresh=inc_ref)[0]
        o1_l2 = avg_sf(32769, w640["l2_meta"], 80.0, 90.0, 1260, 8, n_ph=64, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True, include_refresh=inc_ref)[0]
        o15_l1 = avg_sf(16399, w640["l1_meta"], 80.0, 90.0, 1260, 8, n_ph=64, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True, include_refresh=inc_ref)[0]
        o15_l2 = avg_sf(32783, w640["l2_meta"], 80.0, 90.0, 1260, 8, n_ph=64, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True, include_refresh=inc_ref)[0]
        offset_contrast[lbl] = {
            "dl1": round(o1_l1 - o15_l1, 2),
            "dl2": round(o1_l2 - o15_l2, 2),
            "dtot": round((o1_l1 + o1_l2) - (o15_l1 + o15_l2), 2),
        }

    # Phase sensitivity across 126 phases for 192->640 and 64 phases for 256->960
    phs126 = np.linspace(0, 1260, 126, endpoint=False)
    l1_d = [
        stateful_rtl_model(16385, w640["l1_meta"], 80.0, 90.0, 1260, 8, ph, 3.0, 4.0, 1.0, 4, 4, True, True)[0]
        - stateful_rtl_model(16399, w640["l1_meta"], 80.0, 90.0, 1260, 8, ph, 3.0, 4.0, 1.0, 4, 4, True, True)[0]
        for ph in phs126
    ]
    l2_d = [
        stateful_rtl_model(32769, w640["l2_meta"], 80.0, 90.0, 1260, 8, ph, 3.0, 4.0, 1.0, 4, 4, True, True)[0]
        - stateful_rtl_model(32783, w640["l2_meta"], 80.0, 90.0, 1260, 8, ph, 3.0, 4.0, 1.0, 4, 4, True, True)[0]
        for ph in phs126
    ]
    w960 = wl_map["ffn_256_960_256"]
    ph_sens_960 = {}
    for sd in [60, 90, 100]:
        ri = sd * 14
        vals = [
            stateful_rtl_model(0, w960["l1_meta"], 80.0, float(sd), ri, 8, ph, 3.0, 4.0, 0.0, 4, 4, True, False)[0]
            + stateful_rtl_model(0, w960["l2_meta"], 80.0, float(sd), ri, 8, ph, 3.0, 4.0, 0.0, 4, 4, True, False)[0]
            for ph in np.linspace(0, ri, 64, endpoint=False)
        ]
        ph_sens_960[str(sd)] = {
            "phase_sd_cyc": round(float(np.std(vals, ddof=1)), 2),
            "range_cyc": [round(float(min(vals)), 1), round(float(max(vals)), 1)],
        }
    phase_sensitivity = {
        "offset_192_640_80_90": {
            "dl1_sd": round(float(np.std(l1_d, ddof=1)), 2),
            "dl1_range": [round(float(min(l1_d)), 1), round(float(max(l1_d)), 1)],
            "dl2_sd": round(float(np.std(l2_d, ddof=1)), 2),
            "dl2_range": [round(float(min(l2_d)), 1), round(float(max(l2_d)), 1)],
        },
        "shape_256_960": ph_sens_960,
    }

    # 32-offset sweeps (192->640 and 192->512 @ 80/90 MHz)
    p3 = json.loads((BUILD / "deep_rtl_validation_part3.json").read_text())
    c14 = json.loads((BUILD / "c1_c4_silicon_results.json").read_text())
    w512 = make_synthetic_integer_ffn(192, 512, seed=20261121)
    offset32_comparison = {}
    rows640_out = []
    for label, rows, wl, save_rows in [
        ("192->640->192", p3["offset_0_31_n10_90mhz"]["rows"], w640, True),
        ("192->512->192", c14["c3_offset_192_512_90mhz"]["rows"], w512, False),
    ]:
        meas = np.array([r["meas_tot_mean"] if "meas_tot_mean" in r else r["meas_mean"] for r in rows], float)
        dc_p = np.array([r["pred_tot"] if "pred_tot" in r else r["pred_total_cyc"] for r in rows], float)
        sf_rtl = np.array([
            avg_sf(16384 + r["offset"], wl["l1_meta"], 80.0, 90.0, 1260, 8, n_ph=64, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True)[0]
            + avg_sf(32768 + r["offset"], wl["l2_meta"], 80.0, 90.0, 1260, 8, n_ph=64, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True)[0]
            for r in rows
        ], float)
        if save_rows:
            for r, sfp in zip(rows, sf_rtl):
                rows640_out.append({
                    "offset": int(r["offset"]),
                    "meas_tot_mean": float(r["meas_tot_mean"]),
                    "pred_dc": float(r["pred_tot"]),
                    "pred_sf_rtl": round(float(sfp), 2),
                })
        comp_dict = {}
        for nm, pr in [("duty_cycle", dc_p), ("stateful_rtl_exact", sf_rtl)]:
            sl, ic = np.polyfit(pr, meas, 1)
            rr = float(np.corrcoef(pr, meas)[0, 1])
            n = len(meas)
            yhat = sl * pr + ic
            se_sl = math.sqrt(float(np.sum((meas - yhat) ** 2)) / (n - 2) / float(np.sum((pr - pr.mean()) ** 2)))
            comp_dict[nm] = {
                "slope": round(float(sl), 4),
                "slope_ci95": [round(float(sl - 2.042 * se_sl), 4), round(float(sl + 2.042 * se_sl), 4)],
                "r": round(rr, 4),
                "mae_cyc": round(float(np.mean(np.abs(pr - meas))), 2),
                "bias_cyc": round(float(np.mean(pr - meas)), 2),
            }
        offset32_comparison[label] = comp_dict

    # Runtime benchmark on 192->640->192 (80/90 MHz)
    t0 = time.perf_counter()
    for _ in range(2000):
        _ = predict_rtl_gen(0, w640["l1_meta"], 4, 80.0, 90.0, 1260, 8) + predict_rtl_gen(4160, w640["l2_meta"], 4, 80.0, 90.0, 1260, 8)
    t_cf_us = (time.perf_counter() - t0) / 2000.0 * 1e6

    t0 = time.perf_counter()
    for _ in range(500):
        _ = stateful_rtl_model(16385, w640["l1_meta"], 80.0, 90.0, 1260, 8, 630.0)
        _ = stateful_rtl_model(32769, w640["l2_meta"], 80.0, 90.0, 1260, 8, 630.0)
    t_sf1_us = (time.perf_counter() - t0) / 500.0 * 1e6

    t0 = time.perf_counter()
    for _ in range(40):
        _ = avg_sf(16385, w640["l1_meta"], 80.0, 90.0, 1260, 8, n_ph=32)
        _ = avg_sf(32769, w640["l2_meta"], 80.0, 90.0, 1260, 8, n_ph=32)
    t_sf32_ms = (time.perf_counter() - t0) / 40.0 * 1e3

    runtime_bench = {
        "closed_form_python_us": round(t_cf_us, 2),
        "stateful_1phase_python_us": round(t_sf1_us, 2),
        "stateful_32phase_python_ms": round(t_sf32_ms, 2),
    }

    # -------------------------------------------------------------------------
    # 2. All 3 WB variants + GB_60 on the 45-condition Held-Out set + 18 Frozen non-1:1
    # -------------------------------------------------------------------------
    fc14 = json.loads((BUILD / "frozen_heldout_predictions_c1_c4.json").read_text())
    e79 = json.loads((BUILD / "e7_e9_silicon_results.json").read_text())["E8_E9_rows"]
    b156 = json.loads((BUILD / "b1_b5_b6_silicon_results.json").read_text())
    gb_60 = b156["frozen_greybox_constants"]
    v43 = json.loads((BUILD / "validation_43_heldout_results.json").read_text())

    def build_val43_items() -> list[dict]:
        t4_shapes = [(128, 256), (320, 960)]
        t8_shapes = [(128, 256), (192, 384), (256, 576), (256, 960), (512, 1536)]
        items = []
        for key, tt, sh_list in [
            ("t4_e80_s100_heldout", 4, t4_shapes),
            ("t4_e70_s100_heldout", 4, t4_shapes),
            ("t8_e70_s100_heldout", 8, t8_shapes),
        ]:
            addr = 0
            for idx, ((dm, df), r) in enumerate(zip(sh_list, v43[key])):
                wl = make_ffn(dm, df, tt, seed=20261160 + idx + tt * 100)
                b1 = addr
                addr += len(wl["l1_words"])
                b2 = addr
                addr += len(wl["l2_words"])
                items.append(dict(
                    suite="Val43", frozen="---", meas_commit="763c00d", t_tile=tt,
                    shape=r["shape"], eng=float(r["eng_nom"]), sd=float(r["sd_nom"]),
                    l1_base=b1, l2_base=b2, l1_meta=wl["l1_meta"], l2_meta=wl["l2_meta"],
                    meas=r["meas_mean"], sd_rep=r["meas_sd"],
                ))
        return items

    heldout_45 = []
    for f_row, m_row in zip(fc14["c1_heldout_extrapolation"], c14["c1_heldout"]["rows"]):
        dm, df = parse_sh(m_row["shape"])
        wl = make_ffn(dm, df, 4, seed=1)
        heldout_45.append(dict(suite="C1", frozen="4bbade7", meas_commit="1589dd0", t_tile=4, shape=m_row["shape"], eng=80.0, sd=100.0, l1_base=f_row["l1_base"], l2_base=f_row["l2_base"], l1_meta=wl["l1_meta"], l2_meta=wl["l2_meta"], meas=m_row["meas_mean"], sd_rep=m_row["meas_sd"]))
    for f_row, m_row in zip(fc14["c2_t8_new_clocks"], c14["c2_t8_new_clocks"]):
        dm, df = parse_sh(m_row["shape"])
        wl = make_ffn(dm, df, 8, seed=1)
        heldout_45.append(dict(suite="C2", frozen="4bbade7", meas_commit="1589dd0", t_tile=8, shape=m_row["shape"], eng=float(m_row["eng_mhz"]), sd=float(m_row["sdram_mhz"]), l1_base=f_row["l1_base"], l2_base=f_row["l2_base"], l1_meta=wl["l1_meta"], l2_meta=wl["l2_meta"], meas=m_row["meas_mean"], sd_rep=m_row["meas_sd"]))
    for r in e79:
        if r["tag"] != "e80_s64":
            continue
        dm, df = parse_sh(r["shape"])
        wl = make_ffn(dm, df, 4, seed=1)
        heldout_45.append(dict(suite="E9", frozen="7aa1438", meas_commit="cfa0540", t_tile=4, shape=r["shape"], eng=float(r["eng_mhz"]), sd=float(r["sdram_mhz"]), l1_base=r["l1_base"], l2_base=r["l2_base"], l1_meta=wl["l1_meta"], l2_meta=wl["l2_meta"], meas=r["meas_mean"], sd_rep=r["meas_sd"]))
    for r in b156["b1_t4_records"]:
        dm, df = parse_sh(r["shape"])
        wl = make_ffn(dm, df, 4, seed=1)
        heldout_45.append(dict(suite="B1_T4", frozen="---", meas_commit="763c00d", t_tile=4, shape=r["shape"], eng=float(r["eng_mhz"]), sd=float(r["sdram_mhz"]), l1_base=r["l1_base"], l2_base=r["l2_base"], l1_meta=wl["l1_meta"], l2_meta=wl["l2_meta"], meas=r["meas_tot_mean"], sd_rep=r["meas_tot_sd"]))
    for r in b156["b1_t8_records"]:
        dm, df = parse_sh(r["shape"])
        wl = make_ffn(dm, df, 8, seed=1)
        heldout_45.append(dict(suite="B1_T8", frozen="---", meas_commit="763c00d", t_tile=8, shape=r["shape"], eng=float(r["eng_mhz"]), sd=float(r["sdram_mhz"]), l1_base=r["l1_base"], l2_base=r["l2_base"], l1_meta=wl["l1_meta"], l2_meta=wl["l2_meta"], meas=r["meas_tot_mean"], sd_rep=r["meas_tot_sd"]))
    heldout_45.extend(build_val43_items())

    heldout_3way = {}
    for grp_name, items in [
        ("c1_frozen_3", [x for x in heldout_45 if x["suite"] == "C1"]),
        ("c2_frozen_12", [x for x in heldout_45 if x["suite"] == "C2"]),
        ("e9_frozen_3", [x for x in heldout_45 if x["suite"] == "E9"]),
        ("frozen_non11_18", [x for x in heldout_45 if x["frozen"] != "---"]),
        ("b1_t4_9", [x for x in heldout_45 if x["suite"] == "B1_T4"]),
        ("b1_t8_9", [x for x in heldout_45 if x["suite"] == "B1_T8"]),
        ("val43_9", [x for x in heldout_45 if x["suite"] == "Val43"]),
        ("pooled_non11_45", heldout_45),
    ]:
        res_v = {}
        for vname in ["base_dc", "sf_dropin", "sf_rtl_exact", "gb_60"]:
            tlist = []
            for x in items:
                ri = int(x["sd"] * 14)
                tr = 9 if x["sd"] > 100 else 8
                tt = x["t_tile"]
                comp = compute_cycles(x["l1_meta"], tt) + compute_cycles(x["l2_meta"], tt)
                if vname == "base_dc":
                    pred = predict_rtl_gen(x["l1_base"], x["l1_meta"], tt, x["eng"], x["sd"], ri, tr) + predict_rtl_gen(x["l2_base"], x["l2_meta"], tt, x["eng"], x["sd"], ri, tr)
                elif vname == "sf_dropin":
                    pred = (
                        avg_sf(x["l1_base"], x["l1_meta"], x["eng"], x["sd"], ri, tr, n_ph=32, t_tile=tt, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=0.0, cdc_absorbs_ref=False)[0]
                        + avg_sf(x["l2_base"], x["l2_meta"], x["eng"], x["sd"], ri, tr, n_ph=32, t_tile=tt, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=0.0, cdc_absorbs_ref=False)[0]
                    )
                elif vname == "sf_rtl_exact":
                    pred = (
                        avg_sf(x["l1_base"], x["l1_meta"], x["eng"], x["sd"], ri, tr, n_ph=32, t_tile=tt, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True)[0]
                        + avg_sf(x["l2_base"], x["l2_meta"], x["eng"], x["sd"], ri, tr, n_ph=32, t_tile=tt, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True)[0]
                    )
                else:
                    def gb_l(b: int, m: dict) -> float:
                        nt, kch, kc = m["n_tiles"], m["k_chunks"], m["k_chunk"]
                        P = 8 + m["w_words_per_chunk"]
                        M = nt * kch
                        ct = [tt * kc + (6.0 if (i % kch) < kch - 1 else 13.0) for i in range(M)]
                        rf = 1.0 + gb_60["t_rfc"] / ri
                        cd = [((8 * P + gb_60["c_page"] * ((b + i * P + P - 1) // 32 - (b + i * P) // 32 + 1) + 1) * rf + gb_60["c_cdc_sd"]) * (x["eng"] / x["sd"]) + gb_60["c_cdc_eng"] for i in range(M)]
                        return sum(ct) + cd[0] + sum(max(0.0, cd[i] - ct[i - 1]) for i in range(1, M))
                    pred = gb_l(x["l1_base"], x["l1_meta"]) + gb_l(x["l2_base"], x["l2_meta"])
                tlist.append((x["meas"], pred, comp, x["sd_rep"]))
            res_v[vname] = summarize_suite(tlist)
        heldout_3way[grp_name] = res_v

    # E8 1:1 frozen (6 points) + T=8 20-core
    e8_50_it, e8_90_it = [], []
    for r in e79:
        dm, df = parse_sh(r["shape"])
        wl = make_ffn(dm, df, 4, seed=1)
        tup = (r["meas_mean"], r["pred_base_wb"], compute_cycles(wl["l1_meta"], 4) + compute_cycles(wl["l2_meta"], 4), r["meas_sd"])
        if r["tag"] == "e50_s50":
            e8_50_it.append(tup)
        elif r["tag"] == "e90_s90":
            e8_90_it.append(tup)

    # T=8 40 MHz core (4 points) WB vs GB_60
    t8_40_wb, t8_40_gb = [], []
    for r in p3["t8_freq_sweep"]:
        if r["mhz"] != 40:
            continue
        dm, df = parse_sh(r["shape"])
        wl = make_ffn(dm, df, 8, seed=1)
        meas = float(r["meas_tot_mean"])
        wb_p = predict_rtl_gen(0, wl["l1_meta"], 8, 80.0, 40.0, 560, 8) + predict_rtl_gen(0, wl["l2_meta"], 8, 80.0, 40.0, 560, 8)
        def gb_l(m: dict) -> float:
            nt, kch, kc = m["n_tiles"], m["k_chunks"], m["k_chunk"]
            P = 8 + m["w_words_per_chunk"]
            M = nt * kch
            ct = [8 * kc + (6.0 if (i % kch) < kch - 1 else 13.0) for i in range(M)]
            rf = 1.0 + gb_60["t_rfc"] / 560.0
            cd = [((8 * P + gb_60["c_page"] * ((i * P + P - 1) // 32 - (i * P) // 32 + 1) + 1) * rf + gb_60["c_cdc_sd"]) * 2.0 + gb_60["c_cdc_eng"] for i in range(M)]
            return sum(ct) + cd[0] + sum(max(0.0, cd[i] - ct[i - 1]) for i in range(1, M))
        gb_p = gb_l(wl["l1_meta"]) + gb_l(wl["l2_meta"])
        t8_40_wb.append(abs(wb_p - meas))
        t8_40_gb.append(abs(gb_p - meas))

    # -------------------------------------------------------------------------
    # 3. Raw repeat autocorrelation & SE_eff
    # -------------------------------------------------------------------------
    p1 = json.loads((BUILD / "deep_rtl_validation_part1.json").read_text())
    r6 = json.loads((BUILD / "round6_silicon_checks.json").read_text())
    raw_series = []
    def find_raw_arrays(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                find_raw_arrays(v)
        elif isinstance(obj, list):
            if len(obj) == 10 and all(isinstance(x, (int, float)) for x in obj):
                raw_series.append(np.array(obj, dtype=float))
            else:
                for v in obj:
                    find_raw_arrays(v)
    find_raw_arrays(p1)
    find_raw_arrays(r6)
    rhos, sds, se_n, se_eff = [], [], [], []
    for arr in raw_series:
        sd = float(np.std(arr, ddof=1))
        sds.append(sd)
        sn = sd / math.sqrt(len(arr))
        se_n.append(sn)
        if sd > 1e-9:
            x0, x1 = arr[:-1] - arr.mean(), arr[1:] - arr.mean()
            rho = max(-0.5, min(0.95, float(np.sum(x0 * x1) / np.sum((arr - arr.mean()) ** 2))))
            rhos.append(rho)
            infl = math.sqrt((1.0 + max(0.0, rho)) / max(1e-6, 1.0 - max(0.0, rho)))
            se_eff.append(sn * infl)
        else:
            se_eff.append(0.0)
    repeat_uncertainty = {
        "n_series": len(raw_series),
        "sd_median_cyc": round(float(np.median(sds)), 2),
        "naive_se_median_cyc": round(float(np.median(se_n)), 2),
        "lag1_rho_median": round(float(np.median(rhos)), 3),
        "lag1_rho_iqr": [round(float(np.percentile(rhos, 25)), 3), round(float(np.percentile(rhos, 75)), 3)],
        "se_eff_median_cyc": round(float(np.median(se_eff)), 2),
        "se_eff_max_cyc": round(float(max(se_eff)), 2),
    }

    # -------------------------------------------------------------------------
    # 4. 60-condition Baselines & Clock Selection
    # -------------------------------------------------------------------------
    oos60 = [r for r in d2 if r["group"] == "oos_60pt"]
    methods = ["bist", "ideal_roofline", "comp_plus_cdma0", "avg_page", "wb", "stateful_dropin", "stateful_rtl_exact"]
    res60 = {m: {"tot_pct": [], "st_abs": [], "st_mape": [], "signed": []} for m in methods}
    grid = {}
    for r in oos60:
        w = wl_map[r["shape"]]
        eng, sd, ri, tr = float(r["eng_mhz"]), float(r["sdram_mhz"]), int(r["ref_interval"]), int(r["t_rfc_nom"])
        b1 = layer_baselines(r["l1_base"], w["l1_meta"], eng, sd, ri, tr)
        b2 = layer_baselines(r["l2_base"], w["l2_meta"], eng, sd, ri, tr)
        comp = b1["comp"] + b2["comp"]
        sil = float(r["meas_tot_mean"])
        sil_st = sil - comp
        sf_di = (
            avg_sf(r["l1_base"], w["l1_meta"], eng, sd, ri, tr, n_ph=32, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=0.0, cdc_absorbs_ref=False)[0]
            + avg_sf(r["l2_base"], w["l2_meta"], eng, sd, ri, tr, n_ph=32, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=0.0, cdc_absorbs_ref=False)[0]
        )
        sf_re = (
            avg_sf(r["l1_base"], w["l1_meta"], eng, sd, ri, tr, n_ph=32, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True)[0]
            + avg_sf(r["l2_base"], w["l2_meta"], eng, sd, ri, tr, n_ph=32, req_cdc_sd=3.0, ack_cdc_eng=4.0, rearm_gap_eng=1.0, cdc_absorbs_ref=True)[0]
        )
        preds = {
            "bist": float(r["bist_pred"]),
            "ideal_roofline": b1["ideal_roofline"] + b2["ideal_roofline"],
            "comp_plus_cdma0": b1["comp_plus_cdma0"] + b2["comp_plus_cdma0"],
            "avg_page": b1["avg_page"] + b2["avg_page"],
            "wb": b1["wb"] + b2["wb"],
            "stateful_dropin": sf_di,
            "stateful_rtl_exact": sf_re,
        }
        grid[(r["shape"], int(sd))] = {"silicon": sil, **preds}
        for m, p in preds.items():
            res60[m]["tot_pct"].append(abs(p - sil) / sil * 100.0)
            res60[m]["st_abs"].append(abs(p - sil))
            res60[m]["st_mape"].append(abs(p - sil) / sil_st * 100.0)
            res60[m]["signed"].append(p - sil)

    def feats_lin(r: dict, wl: dict) -> tuple[np.ndarray, float]:
        m1, m2 = wl["l1_meta"], wl["l2_meta"]
        P1 = 8 + m1["w_words_per_chunk"]
        M1 = m1["n_tiles"] * m1["k_chunks"]
        P2 = 8 + m2["w_words_per_chunk"]
        M2 = m2["n_tiles"] * m2["k_chunks"]
        ratio = float(r["eng_mhz"]) / float(r["sdram_mhz"])
        comp = compute_cycles(m1, 4) + compute_cycles(m2, 4)
        tot_w = (P1 * M1 + P2 * M2) * ratio
        tot_m = (M1 + M2) * ratio
        exc = max(0.0, 8.0 * tot_w - comp)
        return np.array([1.0, tot_w, exc, tot_m]), comp

    shapes6 = sorted(set(r["shape"] for r in oos60))
    lin_tot, lin_abs, lin_mape, lin_signed = [], [], [], []
    for hold_sh in shapes6:
        X_tr, y_tr = [], []
        for r in oos60:
            if r["shape"] == hold_sh:
                continue
            f, comp = feats_lin(r, wl_map[r["shape"]])
            X_tr.append(f)
            y_tr.append(r["meas_tot_mean"] - comp)
        beta, _, _, _ = np.linalg.lstsq(np.array(X_tr), np.array(y_tr), rcond=None)
        for r in oos60:
            if r["shape"] != hold_sh:
                continue
            f, comp = feats_lin(r, wl_map[r["shape"]])
            pred_tot = comp + max(0.0, float(f @ beta))
            sil = float(r["meas_tot_mean"])
            lin_tot.append(abs(pred_tot - sil) / sil * 100.0)
            lin_abs.append(abs(pred_tot - sil))
            lin_mape.append(abs(pred_tot - sil) / (sil - comp) * 100.0)
            lin_signed.append(pred_tot - sil)

    baselines_60 = {
        m: {
            "tot_mean_pct": round(float(np.mean(v["tot_pct"])), 4),
            "tot_med_pct": round(float(np.median(v["tot_pct"])), 4),
            "tot_max_pct": round(float(np.max(v["tot_pct"])), 4),
            "stall_mae_cyc": round(float(np.mean(v["st_abs"])), 1),
            "stall_med_cyc": round(float(np.median(v["st_abs"])), 1),
            "stall_mape_pct": round(float(np.mean(v["st_mape"])), 3),
            "stall_mape_max_pct": round(float(np.max(v["st_mape"])), 3),
            "signed_bias_cyc": round(float(np.mean(v["signed"])), 1),
        }
        for m, v in res60.items()
    }
    baselines_60["linear_3feat_losocv"] = {
        "tot_mean_pct": round(float(np.mean(lin_tot)), 4),
        "tot_med_pct": round(float(np.median(lin_tot)), 4),
        "tot_max_pct": round(float(np.max(lin_tot)), 4),
        "stall_mae_cyc": round(float(np.mean(lin_abs)), 1),
        "stall_med_cyc": round(float(np.median(lin_abs)), 1),
        "stall_mape_pct": round(float(np.mean(lin_mape)), 3),
        "stall_mape_max_pct": round(float(np.max(lin_mape)), 3),
        "signed_bias_cyc": round(float(np.mean(lin_signed)), 1),
    }
    gb_60_stats = stateful_subsets["oos_60"]["grey_box_losocv"]
    baselines_60["grey_box_losocv"] = {
        "tot_mean_pct": gb_60_stats["vs_sil_mean_pct"],
        "tot_max_pct": gb_60_stats["vs_sil_max_pct"],
        "stall_mae_cyc": round(gb_60_stats["stall_mae_cyc"], 1),
        "stall_med_cyc": round(gb_60_stats["stall_med_cyc"], 1),
        "stall_mape_pct": round(gb_60_stats["stall_mape_pct"], 2),
        "stall_mape_max_pct": round(gb_60_stats["stall_mape_max_pct"], 2),
        "signed_bias_cyc": round(gb_60_stats["vs_sil_signed_cyc"], 1),
    }

    freqs10 = sorted(set(sd for _, sd in grid.keys()))
    clock_selection = {}
    for m in ["silicon", "wb", "stateful_dropin", "avg_page", "ideal_roofline", "comp_plus_cdma0", "bist"]:
        picks = {}
        matches = {}
        for thr in [0.5, 1.0, 2.0, 3.0]:
            ok = 0
            for sh in shapes6:
                ref_s = grid[(sh, 120)]["silicon"]
                sil_pick = min(f for f in freqs10 if (grid[(sh, f)]["silicon"] - ref_s) / ref_s * 100.0 <= thr)
                ref_m = grid[(sh, 120)][m]
                m_pick = min(f for f in freqs10 if (grid[(sh, f)][m] - ref_m) / ref_m * 100.0 <= thr)
                if thr == 1.0:
                    pen = (grid[(sh, m_pick)]["silicon"] - ref_s) / ref_s * 100.0
                    picks[sh] = {"pick_mhz": m_pick, "actual_penalty_pct": round(pen, 2)}
                if m_pick == sil_pick:
                    ok += 1
            matches[str(thr)] = ok
        clock_selection[m] = {"matches_of_6": matches, "picks_1pct": picks}

    # RD_LAT 4->6
    p_lat4 = predict_layer_param(0, w640["l1_meta"], 80.0, 100.0, 1400, 14.0, 3.0, 4.0, 8.0) + predict_layer_param(4160, w640["l2_meta"], 80.0, 100.0, 1400, 14.0, 3.0, 4.0, 8.0)
    p_lat6 = predict_layer_param(0, w640["l1_meta"], 80.0, 100.0, 1400, 16.0, 3.0, 4.0, 8.0) + predict_layer_param(4160, w640["l2_meta"], 80.0, 100.0, 1400, 16.0, 3.0, 4.0, 8.0)

    payload = {
        "stateful_subsets": stateful_subsets,
        "heldout_3way": heldout_3way,
        "e8_1to1_frozen": {"50_50": summarize_suite(e8_50_it), "90_90": summarize_suite(e8_90_it)},
        "t8_40mhz_wb_vs_gb60": {
            "n": 4,
            "wb_stall_mae_cyc": round(float(np.mean(t8_40_wb)), 2),
            "gb60_stall_mae_cyc": round(float(np.mean(t8_40_gb)), 2),
        },
        "per_freq_60": per_freq_60,
        "offset_contrast_1_15": offset_contrast,
        "phase_sensitivity": phase_sensitivity,
        "offset32_comparison": offset32_comparison,
        "offset32_640_rows": rows640_out,
        "runtime_bench": runtime_bench,
        "repeat_uncertainty": repeat_uncertainty,
        "baselines_60": baselines_60,
        "clock_selection": clock_selection,
        "rd_lat_intervention": {
            "pred_rd4": round(p_lat4, 2),
            "pred_rd6": round(p_lat6, 2),
            "pred_delta": round(p_lat6 - p_lat4, 2),
            "meas_rd4": p3["generalization_100mhz"]["rd_lat_ablation"]["rd_lat_4"]["mean"],
            "meas_rd6": p3["generalization_100mhz"]["rd_lat_ablation"]["rd_lat_6"]["mean"],
            "meas_delta": round(
                p3["generalization_100mhz"]["rd_lat_ablation"]["rd_lat_6"]["mean"]
                - p3["generalization_100mhz"]["rd_lat_ablation"]["rd_lat_4"]["mean"],
                2,
            ),
        },
    }
    out_path = BUILD / "revision_stateful_refresh_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print("45 Held-out 3-way:", json.dumps(heldout_3way["pooled_non11_45"], indent=2))
    print("18 Frozen non-1:1:", json.dumps(heldout_3way["frozen_non11_18"], indent=2))
    print("T=8 40 MHz WB vs GB60:", payload["t8_40mhz_wb_vs_gb60"])
    print("Repeat uncertainty:", repeat_uncertainty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
