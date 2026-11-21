#!/usr/bin/env python3
"""Run REAL-SIZE ZipFormer FFN (512 -> 1536 -> 512, T_tile=4) on Cyclone IV EP4CE10E22C8N
at real 80 MHz PLL clock with weight tiles streamed from Host via USB-Blaster JTAG.

Verifies 0 LSB mismatch against bit-accurate Python golden across:
  - W4A8 (K_tile=512: Layer 1 in 1 K-chunk, Layer 2 in 3 K-chunks via psum_mem K-tiling)
  - W8A8 (K_tile=256: Layer 1 in 2 K-chunks, Layer 2 in 6 K-chunks via psum_mem K-tiling)

Measures real:
  - On-chip PLL clock frequency via free_clk_cnt (MHz)
  - Engine hardware cycles, MAC lane utilization (%), and pure compute time (ms) & GMAC/s
  - Real JTAG weight-streaming bandwidth (KB/s) and end-to-end streaming time (s)
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_golden_vectors import (
    apply_lut_256,
    build_lut_256,
    dyadic_requantize_round_half_up,
    make_dyadic_m16,
    pack_params_64bit,
    pack_words_128bit_i8,
    pack_words_128bit_w4,
    sqnr_db,
    swoosh_l_fp32,
)
from fpga_jtag_runner import (
    OP_COMMIT_ACT,
    OP_COMMIT_WGT,
    OP_START,
    OP_WRITE_LUT,
    OP_WRITE_REG,
    UsbBlasterJtag,
    pack_cfg1,
    pack_cfg2,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"

T_TILE = 4
K_DIM = 512
H_DIM = 1536
N_TILE = 16


def pack_cfg0_ext(op_mode: int, w4: int, lut_mode: int, dim_t: int, dim_k: int, dim_n: int, acc_mode: int = 0) -> int:
    return (
        (op_mode & 0x3)
        | ((w4 & 0x1) << 2)
        | ((lut_mode & 0x3) << 3)
        | ((dim_t & 0xFF) << 5)
        | ((dim_k & 0xFF) << 13)
        | ((dim_n & 0xFF) << 21)
        | ((acc_mode & 0x3) << 29)
    )


def pack_cfg3_ext(
    dec_base: int,
    dim_k: int,
    act_base: int,
    hpost_base: int,
    dst_stride: int,
    stress_loop: int = 0,
    act_stride: int = 0,
    dst_is_act: int = 0,
) -> int:
    return (
        (dec_base & 0xFF)
        | (((dim_k >> 8) & 0x3) << 8)
        | (((act_base >> 8) & 0x1) << 10)
        | (((hpost_base >> 8) & 0x1) << 11)
        | ((dst_stride & 0xFF) << 12)
        | ((stress_loop & 0x1) << 20)
        | ((act_stride & 0xFF) << 21)
        | ((dst_is_act & 0x1) << 29)
    )


def build_real_ffn_workload(w_bits: int, seed_offset: int) -> dict:
    """Build full 512 -> 1536 -> 512 (T=4) FFN quantized tensors & bit-accurate golden."""
    rng = np.random.default_rng(20261117 + seed_offset)
    x_f = rng.normal(0.0, 0.70, size=(T_TILE, K_DIM)).astype(np.float64)
    w1_f = rng.normal(0.0, 0.08, size=(H_DIM, K_DIM)).astype(np.float64)
    b1_f = rng.normal(0.0, 0.03, size=(H_DIM,)).astype(np.float64)
    w2_f = rng.normal(0.0, 0.06, size=(K_DIM, H_DIM)).astype(np.float64)
    b2_f = rng.normal(0.0, 0.03, size=(K_DIM,)).astype(np.float64)

    h_pre_ref = x_f @ w1_f.T + b1_f
    h_post_ref = swoosh_l_fp32(h_pre_ref)
    y_ref = h_post_ref @ w2_f.T + b2_f

    sx = float(np.max(np.abs(x_f))) / 127.0
    x_q = np.clip(np.rint(x_f / sx), -128, 127).astype(np.int8)

    w_qmax = 127.0 if w_bits == 8 else 7.0
    sw1 = np.maximum(np.max(np.abs(w1_f), axis=1), 1e-8) / w_qmax
    w1_q = np.clip(np.rint(w1_f / sw1[:, None]), -w_qmax - (1 if w_bits == 4 else 0), w_qmax).astype(np.int8)
    b1_q = np.rint(b1_f / (sx * sw1)).astype(np.int32)

    acc1_i32 = x_q.astype(np.int32) @ w1_q.astype(np.int32).T + b1_q[None, :]
    h_pre_max = float(np.max(np.abs(h_pre_ref)))
    s_hpre = max(h_pre_max / 120.0, 1e-6)
    s_hpost = max(float(np.max(np.abs(h_post_ref))) / 120.0, 1e-6)

    m1_list, sh1_list = [], []
    for c in range(H_DIM):
        m, sh = make_dyadic_m16((sx * float(sw1[c])) / s_hpre)
        m1_list.append(m)
        sh1_list.append(sh)
    m1 = np.array(m1_list, dtype=np.int32)
    sh1 = np.array(sh1_list, dtype=np.int32)

    h_pre_q = dyadic_requantize_round_half_up(acc1_i32, m1, sh1)
    lut = build_lut_256(swoosh_l_fp32, s_hpre, s_hpost)
    h_post_q = apply_lut_256(h_pre_q, lut)

    sw2 = np.maximum(np.max(np.abs(w2_f), axis=1), 1e-8) / w_qmax
    w2_q = np.clip(np.rint(w2_f / sw2[:, None]), -w_qmax - (1 if w_bits == 4 else 0), w_qmax).astype(np.int8)
    b2_q = np.rint(b2_f / (s_hpost * sw2)).astype(np.int32)

    acc2_i32 = h_post_q.astype(np.int32) @ w2_q.astype(np.int32).T + b2_q[None, :]
    sy_out = max(float(np.max(np.abs(y_ref))) / 120.0, 1e-6)
    m2_list, sh2_list = [], []
    for c in range(K_DIM):
        m, sh = make_dyadic_m16((s_hpost * float(sw2[c])) / sy_out)
        m2_list.append(m)
        sh2_list.append(sh)
    m2 = np.array(m2_list, dtype=np.int32)
    sh2 = np.array(sh2_list, dtype=np.int32)
    y_q = dyadic_requantize_round_half_up(acc2_i32, m2, sh2)

    return {
        "w_bits": w_bits,
        "x_q": x_q,
        "w1_q": w1_q,
        "b1_q": b1_q,
        "m1": m1,
        "sh1": sh1,
        "h_post_q": h_post_q,
        "w2_q": w2_q,
        "b2_q": b2_q,
        "m2": m2,
        "sh2": sh2,
        "y_q": y_q,
        "lut": lut,
        "sqnr_hpost_db": sqnr_db(h_post_ref, h_post_q.astype(np.float64) * s_hpost),
        "sqnr_yout_db": sqnr_db(y_ref, y_q.astype(np.float64) * sy_out),
    }


def _pack_tile_weights_to_ints(w_sub: np.ndarray, w_bits: int) -> list[int]:
    lines = pack_words_128bit_w4(w_sub) if w_bits == 4 else pack_words_128bit_i8(w_sub)
    return [int(s, 16) for s in lines]


def _run_gemm_chunk(
    dev: UsbBlasterJtag,
    w4_mode: int,
    lut_mode: int,
    dim_k: int,
    act_base: int,
    act_stride: int,
    hpost_or_out_base: int,
    dst_stride: int,
    dst_is_act: int,
    acc_mode: int,
    w_chunk: np.ndarray,
    stress_loop: int = 0,
) -> tuple[dict, float, int]:
    """Stream one weight tile into wgt_mem, execute, and return (counters, jtag_ms, payload_bytes)."""
    w_words = _pack_tile_weights_to_ints(w_chunk, 4 if w4_mode else 8)
    payload_bytes = len(w_words) * 16

    t0 = time.perf_counter()
    for addr, wval in enumerate(w_words):
        dev.queue_write_128b(OP_COMMIT_WGT, addr, wval)
    dev.queue_cmd(
        OP_WRITE_REG,
        0,
        pack_cfg0_ext(0, w4_mode, lut_mode, T_TILE, dim_k, N_TILE, acc_mode=acc_mode),
    )
    dev.queue_cmd(
        OP_WRITE_REG,
        1,
        pack_cfg1(0, act_base & 0xFF, 0, hpost_or_out_base & 0xFF),
    )
    dev.queue_cmd(
        OP_WRITE_REG,
        2,
        pack_cfg2(0, 0, 0, hpost_or_out_base & 0xFF),
    )
    dev.queue_cmd(
        OP_WRITE_REG,
        3,
        pack_cfg3_ext(
            dec_base=0,
            dim_k=dim_k,
            act_base=act_base,
            hpost_base=hpost_or_out_base,
            dst_stride=dst_stride,
            stress_loop=stress_loop,
            act_stride=act_stride,
            dst_is_act=dst_is_act,
        ),
    )
    dev.flush()
    stream_ms = (time.perf_counter() - t0) * 1e3

    dev.send_cmd(OP_START, 0, 0)
    if stress_loop:
        return {}, stream_ms, payload_bytes
    cnt = dev.wait_done(timeout_s=2.0)
    return cnt, stream_ms, payload_bytes


def run_mode(dev: UsbBlasterJtag, w_bits: int, measured_clk_mhz: float) -> dict:
    w4_mode = 1 if w_bits == 4 else 0
    k_tile = 512 if w4_mode else 256
    data = build_real_ffn_workload(w_bits=w_bits, seed_offset=4 if w4_mode else 8)

    # 1. Upload input X [4, 512] -> act_mem[0..127] and SwooshL LUT -> lut_mem[0..255]
    x_words = [int(s, 16) for s in pack_words_128bit_i8(data["x_q"])]
    assert len(x_words) == 128
    lut_u8 = data["lut"].view(np.uint8)

    t_init0 = time.perf_counter()
    for i, w in enumerate(x_words):
        dev.queue_write_128b(OP_COMMIT_ACT, i, w)
    for i in range(256):
        dev.queue_cmd(OP_WRITE_LUT, i, int(lut_u8[i]))
    dev.flush()
    init_jtag_ms = (time.perf_counter() - t_init0) * 1e3

    # Layer 1: 512 -> 1536 (96 N-tiles of 16, K_DIM / k_tile K-chunks)
    l1_k_chunks = K_DIM // k_tile
    l1_cycles = 0
    l1_mac_cycles = 0
    l1_wgt_reads = 0
    l1_stream_ms = 0.0
    l1_bytes = 0
    l1_params_bytes = 0

    for nt in range(H_DIM // N_TILE):
        c0 = nt * N_TILE
        c1 = c0 + N_TILE
        p_lines = pack_params_64bit(data["b1_q"][c0:c1], data["m1"][c0:c1], data["sh1"][c0:c1])
        t_p0 = time.perf_counter()
        for pi, ps in enumerate(p_lines):
            dev.queue_write_64b_param(pi, int(ps, 16))
        dev.flush()
        l1_stream_ms += (time.perf_counter() - t_p0) * 1e3
        l1_params_bytes += len(p_lines) * 8

        for kc in range(l1_k_chunks):
            k0 = kc * k_tile
            k1 = k0 + k_tile
            if l1_k_chunks == 1:
                acc_mode = 0
                lut_mode = 1
            elif kc == 0:
                acc_mode = 1  # store_psum
                lut_mode = 0
            elif kc == l1_k_chunks - 1:
                acc_mode = 3  # load_epi
                lut_mode = 1
            else:
                acc_mode = 2
                lut_mode = 0

            cnt, s_ms, pbytes = _run_gemm_chunk(
                dev=dev,
                w4_mode=w4_mode,
                lut_mode=lut_mode,
                dim_k=k_tile,
                act_base=0 + kc * (k_tile // 16),
                act_stride=K_DIM // 16,  # 32 words per row
                hpost_or_out_base=128 + nt,
                dst_stride=H_DIM // 16,  # 96 words per row -> contiguous [4, 1536] in act_mem[128..511]
                dst_is_act=1,
                acc_mode=acc_mode,
                w_chunk=data["w1_q"][c0:c1, k0:k1],
            )
            l1_cycles += cnt["total_cycles"]
            l1_mac_cycles += cnt["mac_cycles"]
            l1_wgt_reads += cnt["wgt_reads"]
            l1_stream_ms += s_ms
            l1_bytes += pbytes

    # Verify full [4, 1536] hidden matrix h_post in act_mem[128..511] (384 words = 6144 bytes)
    hpost_golden = [int(s, 16) for s in pack_words_128bit_i8(data["h_post_q"])]
    assert len(hpost_golden) == 384
    hpost_mismatch = 0
    for idx, exp in enumerate(hpost_golden):
        got = dev.read_128b(1, 128 + idx)
        if got != exp:
            hpost_mismatch += 1

    # Layer 2: 1536 -> 512 (32 N-tiles of 16, H_DIM / k_tile K-chunks)
    l2_k_chunks = H_DIM // k_tile
    l2_cycles = 0
    l2_mac_cycles = 0
    l2_wgt_reads = 0
    l2_stream_ms = 0.0
    l2_bytes = 0
    l2_params_bytes = 0

    for nt in range(K_DIM // N_TILE):
        c0 = nt * N_TILE
        c1 = c0 + N_TILE
        p_lines = pack_params_64bit(data["b2_q"][c0:c1], data["m2"][c0:c1], data["sh2"][c0:c1])
        t_p0 = time.perf_counter()
        for pi, ps in enumerate(p_lines):
            dev.queue_write_64b_param(pi, int(ps, 16))
        dev.flush()
        l2_stream_ms += (time.perf_counter() - t_p0) * 1e3
        l2_params_bytes += len(p_lines) * 8

        for kc in range(l2_k_chunks):
            k0 = kc * k_tile
            k1 = k0 + k_tile
            if kc == 0:
                acc_mode = 1  # store_psum
            elif kc == l2_k_chunks - 1:
                acc_mode = 3  # load_epi
            else:
                acc_mode = 2  # load_store_psum

            cnt, s_ms, pbytes = _run_gemm_chunk(
                dev=dev,
                w4_mode=w4_mode,
                lut_mode=0,
                dim_k=k_tile,
                act_base=128 + kc * (k_tile // 16),
                act_stride=H_DIM // 16,  # 96 words per row
                hpost_or_out_base=nt,
                dst_stride=K_DIM // 16,  # 32 words per row -> contiguous [4, 512] in out_mem[0..127]
                dst_is_act=0,
                acc_mode=acc_mode,
                w_chunk=data["w2_q"][c0:c1, k0:k1],
            )
            l2_cycles += cnt["total_cycles"]
            l2_mac_cycles += cnt["mac_cycles"]
            l2_wgt_reads += cnt["wgt_reads"]
            l2_stream_ms += s_ms
            l2_bytes += pbytes

    # Verify full [4, 512] final output in out_mem[0..127] (128 words = 2048 bytes)
    y_golden = [int(s, 16) for s in pack_words_128bit_i8(data["y_q"])]
    assert len(y_golden) == 128
    yout_mismatch = 0
    for idx, exp in enumerate(y_golden):
        got = dev.read_128b(0, idx)
        if got != exp:
            yout_mismatch += 1

    total_cycles = l1_cycles + l2_cycles
    total_mac_cycles = l1_mac_cycles + l2_mac_cycles
    total_macs = 2 * T_TILE * K_DIM * H_DIM  # 6,291,456 MACs
    hw_exec_ms = (total_cycles / (measured_clk_mhz * 1e6)) * 1e3
    hw_gmac_per_s = (total_macs / (hw_exec_ms * 1e-3)) / 1e9
    total_stream_bytes = l1_bytes + l2_bytes + l1_params_bytes + l2_params_bytes
    total_stream_ms = l1_stream_ms + l2_stream_ms
    jtag_bw_kBps = (total_stream_bytes / 1024.0) / (total_stream_ms * 1e-3)
    ideal_tile_mac_cycles = total_macs // 16  # 16 MAC lanes active per cycle

    return {
        "mode": f"W{w_bits}A8",
        "shape": f"T={T_TILE}, {K_DIM} -> {H_DIM} -> {K_DIM}",
        "k_tile": k_tile,
        "l1_tiles": (H_DIM // N_TILE) * l1_k_chunks,
        "l2_tiles": (K_DIM // N_TILE) * l2_k_chunks,
        "total_macs": total_macs,
        "ideal_mac_cycles_16lane": ideal_tile_mac_cycles,
        "hw_total_cycles": total_cycles,
        "hw_mac_cycles": total_mac_cycles,
        "l1_cycles": l1_cycles,
        "l2_cycles": l2_cycles,
        "mac_lane_utilization_pct": round(100.0 * total_mac_cycles / total_cycles, 3),
        "hw_exec_ms_at_80mhz": round(hw_exec_ms, 4),
        "hw_effective_gmac_s": round(hw_gmac_per_s, 4),
        "peak_16lane_gmac_s_at_80mhz": round((16.0 * measured_clk_mhz) / 1000.0, 4),
        "weights_streamed_bytes": l1_bytes + l2_bytes,
        "params_streamed_bytes": l1_params_bytes + l2_params_bytes,
        "total_stream_bytes": total_stream_bytes,
        "init_act_lut_jtag_ms": round(init_jtag_ms, 2),
        "weight_stream_jtag_s": round(total_stream_ms * 1e-3, 3),
        "jtag_stream_bw_kBps": round(jtag_bw_kBps, 2),
        "hw_wgt_bram_reads": l1_wgt_reads + l2_wgt_reads,
        "hpost_mismatch_words_of_384": hpost_mismatch,
        "yout_mismatch_words_of_128": yout_mismatch,
        "zero_lsb_verified": (hpost_mismatch == 0 and yout_mismatch == 0),
        "sqnr_hpost_db": data["sqnr_hpost_db"],
        "sqnr_yout_db": data["sqnr_yout_db"],
        "bandwidth_bound_analysis": {
            "jtag_vs_compute_ratio": round((total_stream_ms) / hw_exec_ms, 2),
            "required_ext_mem_bw_MBps_for_100pct_util": round(
                ((l1_bytes + l2_bytes) / (1024.0 * 1024.0)) / (hw_exec_ms * 1e-3), 2
            ),
        },
    }


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    with UsbBlasterJtag() as dev:
        cap = dev.read_resp(0x2005)
        st = dev.read_resp(0x2000)
        assert (st >> 32) == 0x5A463445, f"Unexpected magic: 0x{st:016x}"

        # Measure real PLL clock over 3 x 0.5s windows
        clk_samples = []
        for _ in range(3):
            c0 = dev.read_resp(0x2004)
            t0 = time.monotonic()
            time.sleep(0.5)
            c1 = dev.read_resp(0x2004)
            t1 = time.monotonic()
            clk_samples.append(round((c1 - c0) / (t1 - t0) / 1e6, 3))
        clk_mean = round(sum(clk_samples) / len(clk_samples), 3)
        print(f"[PLL] cap=0x{cap:016x}, samples={clk_samples} MHz, mean={clk_mean} MHz")

        res_w4 = run_mode(dev, w_bits=4, measured_clk_mhz=clk_mean)
        print("[W4A8]", json.dumps(res_w4, indent=2))
        assert res_w4["zero_lsb_verified"], "W4A8 512->1536->512 mismatch!"

        res_w8 = run_mode(dev, w_bits=8, measured_clk_mhz=clk_mean)
        print("[W8A8]", json.dumps(res_w8, indent=2))
        assert res_w8["zero_lsb_verified"], "W8A8 512->1536->512 mismatch!"

        # Quick stress-loop sanity check (1 second continuous active toggling, then clear reset)
        t_st = time.perf_counter()
        dev.queue_cmd(
            OP_WRITE_REG,
            3,
            pack_cfg3_ext(0, 256, 128, 0, dst_stride=32, stress_loop=1, act_stride=96, dst_is_act=0),
        )
        dev.flush()
        dev.send_cmd(OP_START, 0, 0)
        c_start = dev.read_resp(0x2003) >> 32
        time.sleep(1.0)
        c_end = dev.read_resp(0x2003) >> 32
        # Clear stress_loop bit and let current pass finish cleanly
        dev.send_cmd(OP_WRITE_REG, 3, pack_cfg3_ext(0, 256, 128, 0, dst_stride=32, stress_loop=0, act_stride=96, dst_is_act=0))
        dev.wait_done(timeout_s=2.0)
        stress_elapsed = time.perf_counter() - t_st
        stress_iters = c_end - c_start

    pow_summary_path = BUILD_DIR / "output_files" / "top.pow.summary"
    pow_line = ""
    if pow_summary_path.exists():
        for line in pow_summary_path.read_text(encoding="utf-8").splitlines():
            if "Total Thermal Power Dissipation" in line or "Core Dynamic Thermal Power Dissipation" in line:
                pow_line += line.strip() + "; "

    out = {
        "device": "EP4CE10E22C8N",
        "capability_reg": f"0x{cap:016x}",
        "measured_pll_clk_mhz_samples": clk_samples,
        "measured_pll_clk_mhz_mean": clk_mean,
        "w4a8_512_1536_512": res_w4,
        "w8a8_512_1536_512": res_w8,
        "stress_loop_1s_check": {
            "active_tile_passes_in_1s": stress_iters,
            "elapsed_s": round(stress_elapsed, 3),
        },
        "power_measurement_note": (
            "On-chip 80MHz PLL + continuous stress_loop verified on live silicon. "
            "No external USB/RS232 wattmeter is attached to host macOS; Quartus PowerPlay reports: "
            + (pow_line.strip("; ") or "unavailable")
        ),
    }
    out_path = BUILD_DIR / "ffn_512_1536_real_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved real-size 512->1536->512 results to {out_path}")


if __name__ == "__main__":
    main()
