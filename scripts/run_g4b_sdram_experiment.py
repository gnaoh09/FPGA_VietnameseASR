#!/usr/bin/env python3
"""G4b Real-Hardware SDR SDRAM (HY57V641620FTP-H, 8 MiB, 16-bit) Experiment.

Tests:
  1. Dual PLL clock verification (80 MHz engine + 80 or 100 MHz SDRAM) & sdram_ready.
  2. Sweep cfg_rd_lat in {2..6} with 64 KiB page-burst BIST.
  3. 1 MiB (65,536 words) and full 8 MiB (524,288 words) OP_BIST verification.
  4. Real-size ZipFormer FFN (512 -> 1536 -> 512, T_tile=4, W4A8) streaming weights
     & params from SDRAM via hardware Ping-Pong DMA, verifying 0 LSB mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpga_jtag_runner import (
    OP_COMMIT_ACT,
    OP_WRITE_REG,
    UsbBlasterJtag,
    pack_cfg1,
    pack_cfg2,
)
from gen_golden_vectors import pack_words_128bit_i8
from run_ffn_512_1536_real import (
    H_DIM,
    K_DIM,
    T_TILE,
    build_real_ffn_workload,
    pack_cfg0_ext,
    pack_cfg3_ext,
)
from sdram_helpers import (
    pack_cfg4,
    pack_layer_w4_sdram_words,
    run_sdram_stream_layer,
    write_lut_256,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"


def measure_pll_clocks(dev: UsbBlasterJtag) -> tuple[float, float, list[float], list[float]]:
    eng_samples, sd_samples = [], []
    for _ in range(3):
        e0 = dev.read_resp(0x2004)
        s0 = dev.read_resp(0x200A)
        t0 = time.perf_counter()
        time.sleep(0.35)
        e1 = dev.read_resp(0x2004)
        s1 = dev.read_resp(0x200A)
        t1 = time.perf_counter()
        dt = t1 - t0
        eng_samples.append(round(((e1 - e0) / dt) / 1e6, 3))
        sd_samples.append(round(((s1 - s0) / dt) / 1e6, 3))
    return (
        round(sum(eng_samples) / len(eng_samples), 3),
        round(sum(sd_samples) / len(sd_samples), 3),
        eng_samples,
        sd_samples,
    )


def pick_optimal_rd_lat(sweep: list[dict]) -> int | None:
    zero_lats = [r["rd_lat"] for r in sweep if r["err_cnt"] == 0]
    if not zero_lats:
        return None
    if 4 in zero_lats and (len(zero_lats) == 1 or (3 in zero_lats and 5 in zero_lats)):
        return 4
    return zero_lats[len(zero_lats) // 2]


def run_bist_stats(dev: UsbBlasterJtag, len_w128: int, sd_mhz: float) -> dict:
    timeout_s = max(10.0, (len_w128 * 20) / (sd_mhz * 1e6) + 5.0)
    res = dev.sdram_run_bist(0, len_w128, timeout_s=timeout_s)
    total_bytes = res["bytes"]
    wr_s = res["wr_cyc"] / (sd_mhz * 1e6)
    rd_s = res["rd_cyc"] / (sd_mhz * 1e6)
    peak_mbps = sd_mhz * 2.0  # 16-bit = 2 B/cycle -> MB/s (1e6 B/s) & MiB/s reported clearly
    wr_mib_s = (total_bytes / (1024.0 * 1024.0)) / wr_s if wr_s > 0 else 0.0
    rd_mib_s = (total_bytes / (1024.0 * 1024.0)) / rd_s if rd_s > 0 else 0.0
    wr_dec_mbps = (total_bytes / 1e6) / wr_s if wr_s > 0 else 0.0
    rd_dec_mbps = (total_bytes / 1e6) / rd_s if rd_s > 0 else 0.0
    return {
        **res,
        "wr_MBps_1e6": round(wr_dec_mbps, 2),
        "rd_MBps_1e6": round(rd_dec_mbps, 2),
        "wr_MiB_s": round(wr_mib_s, 2),
        "rd_MiB_s": round(rd_mib_s, 2),
        "wr_pct_peak": round(100.0 * wr_dec_mbps / peak_mbps, 2),
        "rd_pct_peak": round(100.0 * rd_dec_mbps / peak_mbps, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhz", type=int, choices=[60, 80, 90, 100, 120], required=True, help="Target SDRAM PLL clock MHz")
    args = parser.parse_args()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BUILD_DIR / "g4b_sdram_results.json"

    with UsbBlasterJtag() as dev:
        id_reg = dev.read_resp(0x2000)
        cap_reg = dev.read_resp(0x2005)
        magic = (id_reg >> 32) & 0xFFFFFFFF
        assert magic == 0x5A463445, f"Unexpected FPGA MAGIC 0x{magic:08x}"
        sdram_ready = (cap_reg >> 32) & 1
        ver = cap_reg & 0xFFFF
        print(f"[G4b-{args.mhz}MHz] MAGIC=0x{magic:08x}, VER=0x{ver:04x}, sdram_ready={sdram_ready}")
        assert sdram_ready == 1, "SDRAM controller did not complete init!"

        eng_mhz, sd_mhz, eng_samples, sd_samples = measure_pll_clocks(dev)
        print(f"[PLL] clk_eng={eng_mhz} MHz ({eng_samples}), clk_sdram={sd_mhz} MHz ({sd_samples})")

        # Quick RD128/WR128 sanity check
        test_vec = [0xCAFEBABE_12345678_A5A55A5A_00112233 ^ (i * 0x01010101) for i in range(4)]
        dev.sdram_write_bulk_words(0, test_vec)
        rd_check = [dev.sdram_read_128b(i) for i in range(4)]
        rd_matches_default = sum(1 for a, b in zip(test_vec, rd_check) if a == b)
        print(f"[Probe @ default rd_lat=4] 4-word WR/RD match={rd_matches_default}/4")

        # Sweep cfg_rd_lat in {2..6} using 64 KiB (4,096 words) BIST
        lat_sweep = []
        for lat in range(2, 7):
            dev.send_cmd(OP_WRITE_REG, 4, pack_cfg4(0, 0, 0, lat, 0))
            st = run_bist_stats(dev, 4096, sd_mhz)
            st["rd_lat"] = lat
            lat_sweep.append(st)
            print(f"  [rd_lat={lat}] 64KiB BIST err={st['err_cnt']} rd={st['rd_MBps_1e6']} MB/s wr={st['wr_MBps_1e6']} MB/s")

        best_lat = pick_optimal_rd_lat(lat_sweep)
        if best_lat is None:
            fail_info = {
                "target_sdram_mhz": args.mhz,
                "eng_mhz": eng_mhz,
                "sdram_mhz": sd_mhz,
                "sdram_ready": sdram_ready,
                "rd_lat_sweep": lat_sweep,
                "status": "FAILED_NO_ZERO_ERR_RD_LAT",
            }
            fail_path = BUILD_DIR / f"g4b_probe_fail_{args.mhz}mhz.json"
            fail_path.write_text(json.dumps(fail_info, indent=2), encoding="utf-8")
            print(f"[FAIL] No zero-error rd_lat at {args.mhz} MHz! Wrote {fail_path}")
            return 2

        print(f"[Selected] optimal cfg_rd_lat = {best_lat}")
        dev.send_cmd(OP_WRITE_REG, 4, pack_cfg4(0, 0, 0, best_lat, 0))

        bist_1m = run_bist_stats(dev, 65536, sd_mhz)
        print(f"[1 MiB BIST] err={bist_1m['err_cnt']} wr={bist_1m['wr_MBps_1e6']} MB/s ({bist_1m['wr_pct_peak']}%), rd={bist_1m['rd_MBps_1e6']} MB/s ({bist_1m['rd_pct_peak']}%)")

        bist_8m = run_bist_stats(dev, 524288, sd_mhz)
        print(f"[8 MiB Full Chip BIST] err={bist_8m['err_cnt']} wr={bist_8m['wr_MBps_1e6']} MB/s ({bist_8m['wr_pct_peak']}%), rd={bist_8m['rd_MBps_1e6']} MB/s ({bist_8m['rd_pct_peak']}%)")

        assert bist_1m["err_cnt"] == 0 and bist_8m["err_cnt"] == 0, "SDRAM full BIST had bit errors!"

        # Real-size 512 -> 1536 -> 512 (T=4, W4A8) streaming weights from SDRAM via Ping-Pong DMA
        data = build_real_ffn_workload(w_bits=4, seed_offset=0)
        l1_words, l1_meta = pack_layer_w4_sdram_words(data["w1_q"], data["b1_q"], data["m1"], data["sh1"])
        l2_words, l2_meta = pack_layer_w4_sdram_words(data["w2_q"], data["b2_q"], data["m2"], data["sh2"])
        l1_base = 0
        l2_base = l1_meta["total_words_128b"]
        all_words = l1_words + l2_words

        print(f"[FFN 512->1536->512 W4A8] Uploading {len(all_words)} x 128b words ({len(all_words)*16/1024:.1f} KiB) to SDRAM...")
        up_s = dev.sdram_write_bulk_words(
            l1_base,
            all_words,
            progress_cb=lambda done, tot: print(f"  upload {done}/{tot} ({100.0*done/tot:.0f}%)", flush=True) if done % 8192 == 0 or done == tot else None,
        )
        print(f"  SDRAM one-time upload done in {up_s:.2f} s ({(len(all_words)*16/1024.0)/up_s:.2f} KiB/s)")

        # Upload X_q [4, 512] -> act_mem[0..127] and SwooshL LUT -> lut_mem[0..255]
        x_words = [int(s, 16) for s in pack_words_128bit_i8(data["x_q"])]
        assert len(x_words) == 128
        for idx, w in enumerate(x_words):
            dev.queue_write_128b(OP_COMMIT_ACT, idx, w)
        dev.flush()
        write_lut_256(dev, data["lut"], lut_bank=0)

        # Execute L1 (512 -> 1536, T=4) streaming weights from SDRAM -> act_mem[128..511]
        c_l1 = run_sdram_stream_layer(
            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
            dim_t=T_TILE, k_dim=K_DIM, n_dim=H_DIM, layer_meta=l1_meta,
            sdram_base_w128=l1_base, rd_lat=best_lat, lut_mode=1,
            act_base=0, dst_base=128, dst_is_act=1,
        )
        hpost_golden = [int(s, 16) for s in pack_words_128bit_i8(data["h_post_q"])]
        hpost_hw = [dev.read_128b(1, 128 + i) for i in range(len(hpost_golden))]
        hpost_mism = sum(1 for a, b in zip(hpost_hw, hpost_golden) if a != b)

        # Execute L2 (1536 -> 512, T=4) streaming weights from SDRAM -> out_mem[0..127]
        c_l2 = run_sdram_stream_layer(
            dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
            dim_t=T_TILE, k_dim=H_DIM, n_dim=K_DIM, layer_meta=l2_meta,
            sdram_base_w128=l2_base, rd_lat=best_lat, lut_mode=0,
            act_base=128, dst_base=0, dst_is_act=0,
        )
        y_golden = [int(s, 16) for s in pack_words_128bit_i8(data["y_q"])]
        y_hw = [dev.read_128b(0, i) for i in range(len(y_golden))]
        yout_mism = sum(1 for a, b in zip(y_hw, y_golden) if a != b)

        assert hpost_mism == 0 and yout_mism == 0, f"Mismatch hpost={hpost_mism}, yout={yout_mism}"

        tot_cyc = c_l1["total_cycles"] + c_l2["total_cycles"]
        mac_cyc = c_l1["mac_cycles"] + c_l2["mac_cycles"]
        stall_cyc = c_l1["stall_cycles"] + c_l2["stall_cycles"]
        dma_cyc = c_l1["dma_cycles"] + c_l2["dma_cycles"]
        total_macs = 2 * T_TILE * K_DIM * H_DIM
        exec_ms = (tot_cyc / (eng_mhz * 1e6)) * 1e3
        eff_gmac_s = (total_macs / (exec_ms * 1e-3)) / 1e9
        sdram_bytes_total = l1_meta["total_bytes"] + l2_meta["total_bytes"]
        dma_s = dma_cyc / (eng_mhz * 1e6)
        eff_dma_mbps = (sdram_bytes_total / 1e6) / dma_s if dma_s > 0 else 0.0
        eff_overall_bw_mbps = (sdram_bytes_total / 1e6) / (exec_ms * 1e-3) if exec_ms > 0 else 0.0

        ffn_summary = {
            "shape": f"T={T_TILE}, {K_DIM}->{H_DIM}->{K_DIM}",
            "w4a8_sdram_pingpong": True,
            "sdram_upload_once_s": round(up_s, 3),
            "sdram_resident_bytes": sdram_bytes_total,
            "l1_meta": l1_meta,
            "l2_meta": l2_meta,
            "l1_counters": c_l1,
            "l2_counters": c_l2,
            "total_cycles": tot_cyc,
            "mac_cycles": mac_cyc,
            "dma_cycles": dma_cyc,
            "stall_cycles": stall_cyc,
            "stall_pct_of_total": round(100.0 * stall_cyc / tot_cyc, 2),
            "mac_lane_utilization_pct": round(100.0 * mac_cyc / tot_cyc, 2),
            "hw_exec_ms": round(exec_ms, 4),
            "hw_effective_gmac_s": round(eff_gmac_s, 4),
            "dma_active_bw_MBps_1e6": round(eff_dma_mbps, 2),
            "overall_ffn_sdram_bw_MBps_1e6": round(eff_overall_bw_mbps, 2),
            "hpost_mismatch": hpost_mism,
            "yout_mismatch": yout_mism,
            "zero_lsb_verified": True,
        }
        print("[FFN SDRAM Ping-Pong PASS]", json.dumps({
            k: v for k, v in ffn_summary.items() if k not in ("l1_meta", "l2_meta", "l1_counters", "l2_counters")
        }, indent=2))

        entry = {
            "target_sdram_mhz": args.mhz,
            "measured_eng_mhz": eng_mhz,
            "measured_sdram_mhz": sd_mhz,
            "eng_samples_mhz": eng_samples,
            "sdram_samples_mhz": sd_samples,
            "rd_matches_default_lat4": rd_matches_default,
            "rd_lat_sweep_64kib": lat_sweep,
            "selected_rd_lat": best_lat,
            "bist_1mib": bist_1m,
            "bist_8mib": bist_8m,
            "ffn_512_1536_512_w4a8": ffn_summary,
        }

        all_res = {}
        if out_path.exists():
            all_res = json.loads(out_path.read_text(encoding="utf-8"))
        all_res[f"sdram_{args.mhz}mhz"] = entry
        out_path.write_text(json.dumps(all_res, indent=2), encoding="utf-8")
        print(f"Saved G4b results for {args.mhz} MHz into {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
