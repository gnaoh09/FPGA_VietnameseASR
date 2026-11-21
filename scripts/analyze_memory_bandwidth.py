#!/usr/bin/env python3
"""Quantitative memory, M9K BRAM, and external SDRAM bandwidth feasibility analysis
for Cyclone IV ZipFormer FFN (512 -> 1536 -> 512) at 80 MHz across:
  - T_tile in {4, 8, 16}
  - Weight precision in {W4A8, W8A8}
  - Buffer architecture:
      (A) Standard Activation-Stationary (holds X + h_post + out, double-buffered W4)
      (B) Hidden-Fused Output-Stationary (h_post sliced by N1=16 into tiny 128B reg/M9K,
          accumulates directly into Y_psum INT32 buffer)
  - Target chips / boards:
      - EP4CE6E22C8 (OMDAZZ A2.2, 30 M9K, 0 MB SDRAM U6 unpopulated)
      - EP4CE10E22C8 (46 M9K, with ext SDRAM)
      - EP4CE22F17C6 (Terasic DE0-Nano, 66 M9K, 32 MB 16-bit SDRAM onboard @ 100-143 MHz)
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"

K_DIM = 512
H_DIM = 1536
CLK_MHZ = 80.0
NUM_LANES = 16


def main() -> None:
    scenarios = []
    # Fixed auxiliary M9Ks used by the engine (even before scaling T_tile):
    # param_mem (128x53 -> 2 M9K), lut_mem (512x8 -> 1 M9K), psum_mem (64x32 -> 1 M9K), shift_taps (2 M9K)
    aux_m9k = 6

    for arch in ["standard_hpost_buf", "hidden_fused_y_psum"]:
        for t_tile in [4, 8, 16]:
            for w_bits in [4, 8]:
                bytes_per_elem_w = 0.5 if w_bits == 4 else 1.0
                # Weight buffer: 1 tile of 16 x 512 (or 256 for W8) = 4 KiB = 4 M9K
                # Double-buffered weight prefetch (ping-pong DMA) = 8 M9K
                wgt_single_m9k = 4
                wgt_double_m9k = 8

                if arch == "standard_hpost_buf":
                    # Must store X[T, 512] INT8 + h_post[T, 1536] INT8 (in act_mem) + out[T, 512] INT8
                    act_bytes = t_tile * (K_DIM + H_DIM)  # T * 2048 bytes
                    out_bytes = t_tile * K_DIM            # T * 512 bytes
                    # Each M9K configured 512 x 16/18/32 holds 1 KiB effective at 128-bit width -> 8 M9K per 8 KiB
                    # Specifically, act_mem width 128-bit uses 8 M9K for depth<=512 (8 KiB), 16 M9K for 513..1024 (16 KiB), 32 M9K for 1025..2048 (32 KiB)
                    act_words = act_bytes // 16
                    act_m9k = 8 if act_words <= 512 else (16 if act_words <= 1024 else 32)
                    out_words = out_bytes // 16
                    out_m9k = 4 if out_words <= 512 else 8
                    aux_for_arch = aux_m9k
                    total_weights_bytes = int((K_DIM * H_DIM + H_DIM * K_DIM) * bytes_per_elem_w)
                else:
                    # Hidden-fused: process N1=16 hidden cols at a time!
                    # h_post slice [T, 16] is tiny (T * 16 bytes <= 256 B -> 1 M9K or FF),
                    # while accumulating into Y_psum[T, 512] INT32 (T * 512 * 4 bytes).
                    # Reuse out_m9k=0 (final requantized Y streams/overwrites), but need X[T, 512] + Y_psum[T, 512]
                    act_bytes = t_tile * K_DIM            # X only (T * 512 bytes)
                    out_bytes = t_tile * K_DIM * 4        # Y_psum INT32! (4 bytes per element)
                    act_words = act_bytes // 16
                    act_m9k = 4 if act_words <= 256 else (8 if act_words <= 512 else 16)
                    # Y_psum stored as 4x32b or 16-lane: each M9K is 256x32b = 1 KiB
                    psum_m9k = max(4, (out_bytes + 1023) // 1024)
                    out_m9k = psum_m9k
                    aux_for_arch = 5  # separate psum_mem(64x32) merged into Y_psum
                    total_weights_bytes = int((K_DIM * H_DIM + H_DIM * K_DIM) * bytes_per_elem_w)

                macs = 2 * t_tile * K_DIM * H_DIM
                ideal_cycles = macs // NUM_LANES
                exec_ms = (ideal_cycles / (CLK_MHZ * 1e6)) * 1e3
                req_bw_mbps = (total_weights_bytes / (1024.0 * 1024.0)) / (exec_ms * 1e-3)

                m9k_single_wgt = aux_for_arch + wgt_single_m9k + act_m9k + out_m9k
                m9k_double_wgt = aux_for_arch + wgt_double_m9k + act_m9k + out_m9k

                # Realistic sustained 16-bit SDR SDRAM sequential burst BW:
                # 100 MHz x 2B x ~70% eff = 140 MB/s; 133-143 MHz x 2B x ~70% eff = 186-200 MB/s
                fits_sdram_100m_eff = req_bw_mbps <= 140.0
                fits_de0_nano_sdram = req_bw_mbps <= 190.0

                scenarios.append({
                    "arch": arch,
                    "t_tile": t_tile,
                    "w_bits": w_bits,
                    "weight_traffic_KiB": total_weights_bytes // 1024,
                    "compute_ms_80mhz": round(exec_ms, 3),
                    "required_ext_bw_MBps": round(req_bw_mbps, 2),
                    "m9k_single_wgt": m9k_single_wgt,
                    "m9k_double_wgt_dma": m9k_double_wgt,
                    "fits_ep4ce6_30_m9k_single": m9k_single_wgt <= 30,
                    "fits_ep4ce6_30_m9k_dma": m9k_double_wgt <= 30,
                    "fits_ep4ce10_46_m9k_dma": m9k_double_wgt <= 46,
                    "fits_de0nano_66_m9k_dma": m9k_double_wgt <= 66,
                    "fits_sdram_16b_100mhz_140MBps": fits_sdram_100m_eff,
                    "fits_de0nano_sdram_190MBps": fits_de0_nano_sdram,
                })

    # Also compute JTAG-bound vs SDRAM-assumed full Zipformer-30M RTF (1.2500 GMAC @ T=500, 5.0 s audio)
    gmac_30m_t500 = 1.2500
    w30m_ffn_w4_mb = 16.99  # MiB
    jtag_measured_kbps = 27.74
    jtag_stream_only_s = (w30m_ffn_w4_mb * 1024.0) / jtag_measured_kbps
    jtag_rtf_t500 = jtag_stream_only_s / 5.0

    # EP4CE6 (512->1536->512 real measured rate 1.2733 GMAC/s) with zero-stall memory:
    compute_only_30m_s = (gmac_30m_t500 * 1e9) / (1.2733e9)
    compute_only_rtf = compute_only_30m_s / 5.0

    out = {
        "omdazz_ep4ce6_u6_note": (
            "OMDAZZ / RZ-EasyFPGA A2.2 has unpopulated U6 TSOP-54 16-bit SDR SDRAM footprint "
            "(EP4CE6E22C8 EQFP-144 has 91 total I/O; current bitstream uses 10/92 pins, leaving 82 I/O pins "
            "which suffices for 39-pin 16-bit SDRAM wiring if U6 traces are populated, though double-buffered "
            "T_tile>=8 DMA exceeds EP4CE6's 30 M9K budget)."
        ),
        "rtf_comparison_30m_t500": {
            "measured_jtag_stream_bw_kBps": jtag_measured_kbps,
            "model_w8a8_ffn_w4_weights_MiB": w30m_ffn_w4_mb,
            "real_board_with_jtag_stream_s": round(jtag_stream_only_s, 1),
            "real_board_with_jtag_rtf": round(jtag_rtf_t500, 1),
            "ideal_ext_mem_assumed_compute_s": round(compute_only_30m_s, 3),
            "ideal_ext_mem_assumed_rtf": round(compute_only_rtf, 3),
        },
        "tiling_feasibility_matrix": scenarios,
    }
    out_path = BUILD_DIR / "memory_bandwidth_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
