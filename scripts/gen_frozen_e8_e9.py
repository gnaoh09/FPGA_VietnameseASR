#!/usr/bin/env python3
"""Freeze pre-measurement predictions for E8 (true held-out 1:1 clock pairs: 50/50, 70/70, 90/90 MHz)
and E9 (rational 5:4 ratio 80/64 MHz) on 3 unseen held-out shapes (128->384->128, 320->960->320, 384->1152->384).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_43_silicon_heldout import make_ffn, predict_rtl_gen

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OUT_JSON = BUILD_DIR / "frozen_e8_e9_predictions.json"


def main() -> int:
    shapes = [(128, 384), (320, 960), (384, 1152)]
    wls = []
    addr = 0
    for idx, (dm, df) in enumerate(shapes):
        wl = make_ffn(dm, df, 4, seed=20261250 + idx)
        b1 = addr; addr += len(wl["l1_words"])
        b2 = addr; addr += len(wl["l2_words"])
        M_tot = wl["l1_meta"]["n_tiles"] * wl["l1_meta"]["k_chunks"] + wl["l2_meta"]["n_tiles"] * wl["l2_meta"]["k_chunks"]
        wls.append({"d_model": dm, "d_ff": df, "shape": f"{dm}->{df}->{dm}", "l1_base": b1, "l2_base": b2, "M_tot": M_tot, "wl": wl})

    cases = [
        {"tag": "e50_s50", "svf": "top_eng50_sd50.svf", "eng_mhz": 50, "sdram_mhz": 50, "group": "E8_heldout_1to1"},
        {"tag": "e70_s70", "svf": "top_eng70_sd70.svf", "eng_mhz": 70, "sdram_mhz": 70, "group": "E8_heldout_1to1"},
        {"tag": "e90_s90", "svf": "top_eng90_sd90.svf", "eng_mhz": 90, "sdram_mhz": 90, "group": "E8_heldout_1to1"},
        {"tag": "e80_s64", "svf": "top_sdram64.svf",    "eng_mhz": 80, "sdram_mhz": 64, "group": "E9_rational_5to4"},
    ]

    preds = []
    for c in cases:
        ref_int = c["sdram_mhz"] * 14
        t_rfc = 9 if c["sdram_mhz"] > 100 else 8
        is_1to1 = (c["eng_mhz"] == c["sdram_mhz"])
        for w in wls:
            p1 = predict_rtl_gen(w["l1_base"], w["wl"]["l1_meta"], 4, float(c["eng_mhz"]), float(c["sdram_mhz"]), ref_int, t_rfc)
            p2 = predict_rtl_gen(w["l2_base"], w["wl"]["l2_meta"], 4, float(c["eng_mhz"]), float(c["sdram_mhz"]), ref_int, t_rfc)
            base_pred = p1 + p2
            ext_b3_pred = base_pred + (1.0 * w["M_tot"] if is_1to1 else 0.0)
            preds.append({
                "tag": c["tag"], "svf": c["svf"], "group": c["group"],
                "eng_mhz": c["eng_mhz"], "sdram_mhz": c["sdram_mhz"],
                "shape": w["shape"], "d_model": w["d_model"], "d_ff": w["d_ff"],
                "l1_base": w["l1_base"], "l2_base": w["l2_base"], "M_tot": w["M_tot"],
                "pred_base_wb": round(base_pred, 2),
                "pred_ext_b3": round(ext_b3_pred, 2),
            })

    OUT_JSON.write_text(json.dumps({"predictions": preds}, indent=2), encoding="utf-8")
    print(f"Saved frozen E8/E9 predictions ({len(preds)} conditions) to {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
