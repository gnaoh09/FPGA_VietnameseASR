#!/usr/bin/env python3
"""Generate and freeze out-of-sample white-box RTL predictions for experiments C1..C4
BEFORE flashing or measuring on silicon.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_43_silicon_heldout import make_ffn, predict_rtl_gen

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OUT_JSON = BUILD_DIR / "frozen_heldout_predictions_c1_c4.json"


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # C1: Truly extrapolated held-out shapes (T_tile=4 @ 80/100 MHz):
    #   128->384->128 (smaller than dev, kch=1,1)
    #   384->1152->384 (larger than dev, kch=1,3)
    #   512->1280->512 (larger than dev, kch=1,4)
    c1_specs = [(128, 384), (384, 1152), (512, 1280)]
    c1_preds = []
    addr = 0
    for idx, (dm, df) in enumerate(c1_specs):
        wl = make_ffn(dm, df, 4, seed=20261201 + idx)
        b1 = addr; addr += len(wl["l1_words"])
        b2 = addr; addr += len(wl["l2_words"])
        p1 = predict_rtl_gen(b1, wl["l1_meta"], 4, 80.0, 100.0, ref_int=1400, t_rfc=8)
        p2 = predict_rtl_gen(b2, wl["l2_meta"], 4, 80.0, 100.0, ref_int=1400, t_rfc=8)
        c1_preds.append({
            "shape": f"{dm}->{df}->{dm}",
            "d_model": dm, "d_ff": df, "t_tile": 4,
            "l1_base": b1, "l2_base": b2,
            "l1_kchunks": wl["l1_meta"]["k_chunks"],
            "l2_kchunks": wl["l2_meta"]["k_chunks"],
            "pred_total_cyc": round(p1 + p2, 2),
        })

    # C2: T_tile=8 held-out with new SDRAM clocks {65, 75, 110} MHz (@ 80 MHz Engine)
    c2_specs = [(192, 384), (256, 576), (256, 960), (512, 1536)]
    c2_preds = []
    t8_addr = 0
    t8_wls = []
    for idx, (dm, df) in enumerate(c2_specs):
        wl = make_ffn(dm, df, 8, seed=20261210 + idx)
        b1 = t8_addr; t8_addr += len(wl["l1_words"])
        b2 = t8_addr; t8_addr += len(wl["l2_words"])
        t8_wls.append((dm, df, wl, b1, b2))
    for sd_mhz in (65, 75, 110):
        t_rfc = 9 if sd_mhz > 100 else 8
        ref_int = sd_mhz * 14
        for dm, df, wl, b1, b2 in t8_wls:
            p1 = predict_rtl_gen(b1, wl["l1_meta"], 8, 80.0, float(sd_mhz), ref_int=ref_int, t_rfc=t_rfc)
            p2 = predict_rtl_gen(b2, wl["l2_meta"], 8, 80.0, float(sd_mhz), ref_int=ref_int, t_rfc=t_rfc)
            c2_preds.append({
                "shape": f"{dm}->{df}->{dm}",
                "d_model": dm, "d_ff": df, "t_tile": 8,
                "eng_mhz": 80, "sdram_mhz": sd_mhz,
                "l1_base": b1, "l2_base": b2,
                "pred_total_cyc": round(p1 + p2, 2),
            })

    # C3: Offset sweep 0..31 on 192->512->192 (T_tile=4 @ 80/90 MHz)
    wl_512 = make_ffn(192, 512, 4, seed=20261220)
    c3_preds = []
    for off in range(32):
        b1, b2 = 16384 + off, 32768 + off
        p1 = predict_rtl_gen(b1, wl_512["l1_meta"], 4, 80.0, 90.0, ref_int=1260, t_rfc=8)
        p2 = predict_rtl_gen(b2, wl_512["l2_meta"], 4, 80.0, 90.0, ref_int=1260, t_rfc=8)
        c3_preds.append({"offset": off, "l1_base": b1, "l2_base": b2, "pred_total_cyc": round(p1 + p2, 2)})

    # C4: DSE selecting minimum f_sdram at T_tile=8 (192->384->192) within <=1% of 100 MHz (or 110 MHz)
    # Using White-Box vs BIST (94% of 2*f_sdram MB/s) across {40, 50, 60, 65, 75, 80, 100, 110} MHz
    wl_dse = make_ffn(192, 384, 8, seed=20261210)
    b1_dse, b2_dse = 0, len(wl_dse["l1_words"])
    bytes_tot = wl_dse["l1_meta"]["total_bytes"] + wl_dse["l2_meta"]["total_bytes"]
    act_cyc = (
        wl_dse["l1_meta"]["n_tiles"] * (8 * wl_dse["l1_meta"]["k_chunk"] + 13)
        + wl_dse["l2_meta"]["n_tiles"] * (8 * wl_dse["l2_meta"]["k_chunk"] + 13)
    )
    c4_table = []
    for sd_mhz in (40, 50, 60, 65, 75, 80, 100, 110):
        t_rfc = 9 if sd_mhz > 100 else 8
        wb = (
            predict_rtl_gen(b1_dse, wl_dse["l1_meta"], 8, 80.0, float(sd_mhz), sd_mhz * 14, t_rfc)
            + predict_rtl_gen(b2_dse, wl_dse["l2_meta"], 8, 80.0, float(sd_mhz), sd_mhz * 14, t_rfc)
        )
        bist_bw = 0.94 * 2.0 * sd_mhz * 1e6
        bist = max(float(act_cyc), (bytes_tot / bist_bw) * 80e6)
        c4_table.append({"sdram_mhz": sd_mhz, "wb_pred_cyc": round(wb, 2), "bist_pred_cyc": round(bist, 2)})
    ref_wb = c4_table[-1]["wb_pred_cyc"]
    ref_bist = c4_table[-1]["bist_pred_cyc"]
    wb_sel = min(x["sdram_mhz"] for x in c4_table if x["wb_pred_cyc"] <= 1.01 * ref_wb)
    bist_sel = min(x["sdram_mhz"] for x in c4_table if x["bist_pred_cyc"] <= 1.01 * ref_bist)

    payload = {
        "c1_heldout_extrapolation": c1_preds,
        "c2_t8_new_clocks": c2_preds,
        "c3_offset_192_512_90mhz": c3_preds,
        "c4_t8_dse_192_384": {
            "ref_110mhz_wb": ref_wb,
            "ref_110mhz_bist": ref_bist,
            "wb_selected_mhz_1pct": wb_sel,
            "bist_selected_mhz_1pct": bist_sel,
            "table": c4_table,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Frozen predictions saved to {OUT_JSON}")
    print(f"C4 DSE (<=1% vs 110 MHz @ T_tile=8): WB selects {wb_sel} MHz, BIST selects {bist_sel} MHz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
