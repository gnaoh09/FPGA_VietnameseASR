#!/usr/bin/env python3
"""G5b Extended Hardware Verification on EP4CE10E22C8N + HY57V641620FTP-H (100 MHz SDRAM):

Part A: Full-sequence T hardware execution (not single T=4 tile only) on VIVOSDEV02_R106.wav
  (T_fbank=263, 504 block-tiles = 1,008 layer launches across all 36 FFN blocks) comparing:
  - Measured single T=4 pass (36 block-tiles)
  - Measured complete utterance T_fbank=263 (504 block-tiles)
  - Normalized T_fbank=500 (5.0 s, 978 block-tiles) clearly separated as scaled

Part B: 50 diverse speaker-stratified VIVOS test utterances (1,800 peak/high-stress tiles)
  + 4 adversarial corner-case stress vectors per block (144 tiles covering INT8 saturation/clamp
  [-128, +127] and 0.5-LSB dyadic rounding boundaries) = 1,944 block-tiles on physical silicon,
  plus dataset-wide full-sequence clamp and dyadic rounding tie statistics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fpga_bitexact_30m_onnx import compute_fbank
from fpga_jtag_runner import (
    OP_COMMIT_ACT,
    OP_START,
    OP_WRITE_REG,
    UsbBlasterJtag,
    pack_cfg1,
    pack_cfg2,
)
from gen_golden_vectors import apply_lut_256, pack_words_128bit_i8
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks
from run_g5b_30m_resident_sdram import BITEXACT_W4_ONNX, DEFAULT_WAV, pack_all_36_ffn_image
from sdram_helpers import pack_cfg4, run_sdram_stream_layer, write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
VIVOS_TEST_WAVES = ROOT_DIR / "data" / "vivos" / "test" / "waves"
T_TILE = 4


def py_hw_checksum_stream(y_q: np.ndarray, n_tiles: int) -> int:
    """Exact bit-accurate match to rtl/zipformer_engine.v hw_out_checksum across (nt, t)."""
    csum = 0
    u8 = y_q.view(np.uint8)
    t_dim = y_q.shape[0]
    for nt in range(n_tiles):
        c0 = nt * 16
        for t in range(t_dim):
            bs = u8[t, c0 : c0 + 16].tobytes()
            w0 = int.from_bytes(bs[0:4], "little")
            w1 = int.from_bytes(bs[4:8], "little")
            w2 = int.from_bytes(bs[8:12], "little")
            w3 = int.from_bytes(bs[12:16], "little")
            rot = ((csum << 1) & 0xFFFFFFFF) | ((csum >> 31) & 1)
            csum = rot ^ w0 ^ w1 ^ w2 ^ w3
    return csum


def dyadic_round_and_stats(acc_i32: np.ndarray, m16: np.ndarray, sh: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """Vectorized RTL-exact (acc * m16 + (1<<(sh-1))) >> sh with clamp [-128, 127], plus boundary counts."""
    m64 = m16.astype(np.int64)[None, :]
    s64 = sh.astype(np.int64)[None, :]
    prod = acc_i32.astype(np.int64) * m64
    half = np.where(s64 > 0, np.left_shift(np.int64(1), np.maximum(s64 - 1, 0)), np.int64(0))
    mask = np.where(s64 > 0, np.left_shift(np.int64(1), np.maximum(s64, 0)) - 1, np.int64(0))
    shifted_unclamped = np.right_shift(prod + half, s64)
    clamped = np.clip(shifted_unclamped, -128, 127).astype(np.int8)

    rem = prod & mask
    has_shift = s64 > 0
    exact_ties_mat = has_shift & (rem == half)
    near_ties_mat = has_shift & (np.abs(rem - half) <= 1)

    stats = {
        "elements": int(acc_i32.size),
        "strict_clamp_pos": int(np.sum(shifted_unclamped > 127)),
        "strict_clamp_neg": int(np.sum(shifted_unclamped < -128)),
        "rail_pos_127": int(np.sum(clamped == 127)),
        "rail_neg_128": int(np.sum(clamped == -128)),
        "exact_half_ties": int(np.sum(exact_ties_mat)),
        "near_half_1lsb": int(np.sum(near_ties_mat)),
    }
    return clamped, stats, np.sum((shifted_unclamped > 127) | (shifted_unclamped < -128), axis=1), np.sum(exact_ties_mat, axis=1)


def pick_stratified_vivos_wavs(waves_dir: Path, target_count: int = 50) -> list[Path]:
    """Pick target_count diverse VIVOS test wavs round-robin across all speaker subdirectories."""
    spk_dirs = sorted(p for p in waves_dir.iterdir() if p.is_dir())
    queues = [sorted(d.glob("*.wav")) for d in spk_dirs]
    chosen: list[Path] = []
    idx = 0
    while len(chosen) < target_count:
        progressed = False
        for q in queues:
            if idx < len(q):
                chosen.append(q[idx])
                progressed = True
                if len(chosen) == target_count:
                    break
        if not progressed:
            break
        idx += 1
    return chosen


def build_probe_session(manifest: list[dict]) -> ort.InferenceSession:
    q_model = onnx.load(str(BITEXACT_W4_ONNX))
    tap_names = [f"_fpga_{m['tag']}_xq" for m in manifest]
    existing_outs = {o.name for o in q_model.graph.output}
    for tn in tap_names:
        if tn not in existing_outs:
            q_model.graph.output.append(helper.make_tensor_value_info(tn, TensorProto.DOUBLE, None))
    probe_path = BUILD_DIR / "_g5b_multi_probe.tmp.onnx"
    onnx.save(q_model, str(probe_path))
    try:
        sess = ort.InferenceSession(str(probe_path), providers=["CPUExecutionProvider"])
    finally:
        probe_path.unlink(missing_ok=True)
    return sess


def verify_or_upload_sdram(dev: UsbBlasterJtag, all_words: list[int], manifest: list[dict]) -> tuple[bool, float]:
    """Spot-check 72 words (first & last word of all 36 blocks); upload full 6.52 MiB only if needed."""
    spot_addrs = []
    for m in manifest:
        l1_start = m["l1_sdram_base_w128"]
        l2_end = m["l2_sdram_base_w128"] + m["l2_meta"]["total_words_128b"] - 1
        spot_addrs.extend([l1_start, l2_end])
    all_ok = all(dev.sdram_read_128b(addr) == all_words[addr] for addr in spot_addrs)
    if all_ok:
        return True, 0.0
    up_s = dev.sdram_write_bulk_words(0, all_words)
    return False, up_s


def run_layer_fast(
    dev: UsbBlasterJtag,
    dim_t: int,
    k_dim: int,
    n_dim: int,
    layer_meta: dict,
    sdram_base_w128: int,
    rd_lat: int,
    lut_mode: int,
    act_base: int,
    dst_base: int,
    dst_is_act: int,
) -> dict:
    """Fast SDRAM stream launcher that reads done, cycles, checksum & stall counters (4 reads)."""
    dev.queue_cmd(OP_WRITE_REG, 0, pack_cfg0_ext(0, 1, lut_mode, dim_t, layer_meta["k_chunk"], 16, acc_mode=0))
    dev.queue_cmd(OP_WRITE_REG, 1, pack_cfg1(16, act_base & 0xFF, 0, dst_base & 0xFF))
    dev.queue_cmd(OP_WRITE_REG, 2, pack_cfg2(0, 0, 0, dst_base & 0xFF))
    dev.queue_cmd(
        OP_WRITE_REG,
        3,
        pack_cfg3_ext(
            0,
            layer_meta["k_chunk"],
            act_base,
            dst_base,
            dst_stride=n_dim // 16,
            stress_loop=0,
            act_stride=k_dim // 16,
            dst_is_act=dst_is_act,
        ),
    )
    dev.queue_cmd(OP_WRITE_REG, 4, pack_cfg4(1, layer_meta["n_tiles"], layer_meta["k_chunks"], rd_lat, layer_meta["w_words_per_chunk"]))
    dev.queue_cmd(OP_WRITE_REG, 5, sdram_base_w128 & 0x7FFFF)
    dev.queue_cmd(OP_START, 0, 0)
    dev.flush()

    t0 = time.monotonic()
    while True:
        st = dev.read_resp(0x2000)
        if (st >> 1) & 1:
            break
        if time.monotonic() - t0 > 5.0:
            raise TimeoutError(f"FPGA stream layer timeout (status=0x{st:016x})")
    c1 = dev.read_resp(0x2001)
    c3 = dev.read_resp(0x2003)
    c9 = dev.read_resp(0x2009)
    return {
        "total_cycles": c1 & 0xFFFFFFFF,
        "mac_cycles": (c1 >> 32) & 0xFFFFFFFF,
        "out_checksum": f"0x{c3 & 0xFFFFFFFF:08x}",
        "stall_cycles": (c9 >> 32) & 0xFFFFFFFF,
    }


def eval_block_python(m: dict, x_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict, np.ndarray, np.ndarray]:
    acc1 = x_q.astype(np.int32) @ m["w1_q"].astype(np.int32).T + m["b1_q"][None, :]
    hpre_q, l1_st, l1_row_clamps, l1_row_ties = dyadic_round_and_stats(acc1, m["m1"], m["sh1"])
    hp_q = apply_lut_256(hpre_q, m["lut_i8"])
    acc2 = hp_q.astype(np.int32) @ m["w2_q"].astype(np.int32).T + m["b2_q"][None, :]
    y_q, l2_st, l2_row_clamps, l2_row_ties = dyadic_round_and_stats(acc2, m["m2"], m["sh2"])
    row_score = l1_row_clamps + l2_row_clamps + l1_row_ties + l2_row_ties
    return hpre_q, hp_q, y_q, l1_st, l2_st, row_score, (l1_row_ties + l2_row_ties)


def construct_corner_stress_tiles(m: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate 4 explicit corner-case T=4 tiles per FFN block targeting overflow saturation & 0.5-LSB ties."""
    d_m = m["d_model"]
    s0 = np.where(m["w1_q"][0] >= 0, 127, -128).astype(np.int8)
    s1 = np.where(m["w1_q"][1] >= 0, -128, 127).astype(np.int8)
    tile_saturation = np.stack([
        s0,
        s1,
        np.full(d_m, 127, dtype=np.int8),
        np.full(d_m, -128, dtype=np.int8),
    ])

    alt = np.array([127 if (i & 1) == 0 else -128 for i in range(d_m)], dtype=np.int8)
    tile_checker = np.stack([alt, (-alt - 1).astype(np.int8), np.roll(alt, 1), np.roll((-alt - 1).astype(np.int8), 1)])

    # Search 1,500 random small/medium candidate vectors to maximize exact/near 0.5-LSB rounding boundary hits
    cands = np.vstack([
        rng.integers(-16, 17, size=(700, d_m), dtype=np.int8),
        rng.integers(-128, 128, size=(700, d_m), dtype=np.int8),
        np.eye(min(100, d_m), d_m, dtype=np.int8) * 64,
    ])
    acc1 = cands.astype(np.int32) @ m["w1_q"].astype(np.int32).T + m["b1_q"][None, :]
    prod1 = acc1.astype(np.int64) * m["m1"].astype(np.int64)[None, :]
    s64 = m["sh1"].astype(np.int64)[None, :]
    half1 = np.where(s64 > 0, np.left_shift(np.int64(1), np.maximum(s64 - 1, 0)), np.int64(0))
    mask1 = np.where(s64 > 0, np.left_shift(np.int64(1), np.maximum(s64, 0)) - 1, np.int64(0))
    rem1 = prod1 & mask1
    exact_cnt = np.sum((s64 > 0) & (rem1 == half1), axis=1)
    near_cnt = np.sum((s64 > 0) & (np.abs(rem1 - half1) <= 1), axis=1)
    top_idx = np.argsort(-(exact_cnt * 100 + near_cnt))[:T_TILE]
    tile_ties = cands[top_idx].astype(np.int8)

    alt1 = np.array([1 if (i & 1) == 0 else -1 for i in range(d_m)], dtype=np.int8)
    tile_small = np.stack([
        np.zeros(d_m, dtype=np.int8),
        np.ones(d_m, dtype=np.int8),
        np.full(d_m, -1, dtype=np.int8),
        alt1,
    ])

    return {
        "stress_max_rail_saturation": tile_saturation,
        "stress_checkerboard_extremes": tile_checker,
        "stress_halfway_rounding_ties": tile_ties,
        "stress_zero_and_single_lsb": tile_small,
    }


def _accum_stats(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def run_part_a_full_sequence(dev: UsbBlasterJtag, sess: ort.InferenceSession, manifest: list[dict], rd_lat: int) -> dict:
    """Execute COMPLETE utterance sequence T across all 36 FFN blocks on VIVOSDEV02_R106.wav."""
    audio, sr = sf.read(str(DEFAULT_WAV), dtype="float32")
    audio_dur_s = len(audio) / float(sr)
    feats = compute_fbank(audio, sr)
    outs = sess.run(None, {"x": feats[None], "x_lens": np.array([len(feats)], np.int64)})
    out_map = {o.name: outs[i] for i, o in enumerate(sess.get_outputs())}

    # Also query canonical T_fbank=500 shapes for apples-to-apples comparison
    outs500 = sess.run(None, {"x": np.zeros((1, 500, 80), np.float32), "x_lens": np.array([500], np.int64)})
    sh500 = {o.name: list(r.shape) for o, r in zip(sess.get_outputs(), outs500)}

    tot_cyc_all = 0
    mac_cyc_all = 0
    dma_cyc_all = 0
    stall_cyc_all = 0
    tot_macs_all = 0
    total_tiles_run = 0
    total_layer_launches = 0
    total_sdram_bytes_read = 0
    total_act_in_bytes = 0
    total_act_out_bytes = 0
    total_readback_bytes = 0
    total_readback_words = 0
    lut_upload_s = 0.0
    x_upload_s = 0.0
    hw_wait_exec_s = 0.0
    y_readback_s = 0.0
    csum_mismatches = 0
    byte_mismatches = 0
    stage_agg: dict[str, dict] = {}
    t_utt_by_stage: dict[str, int] = {}

    t500_macs_total = 0
    t500_act_bytes_total = 0
    t500_sdram_bytes_total = 0
    t500_scaled_cyc_total = 0.0
    t500_tiles_total = 0

    for m in manifest:
        pfx = f"_fpga_{m['tag']}_"
        d_m, d_ff = m["d_model"], m["d_ff"]
        xq_full = np.rint(out_map[pfx + "xq"]).astype(np.int8).reshape(-1, d_m)
        _, _, yq_full, _, _, _, _ = eval_block_python(m, xq_full)
        t_real = xq_full.shape[0]
        assert t_real % T_TILE == 0, f"Expected multiple of {T_TILE}, got {t_real}"
        n_t_tiles = t_real // T_TILE
        parts = m["prefix"].strip("/").split("/")
        stage_id = f"stage_{parts[1]}" if len(parts) >= 2 else "?"
        t_utt_by_stage[stage_id] = t_real

        t_lut0 = time.perf_counter()
        write_lut_256(dev, m["lut_i8"], lut_bank=0)
        lut_upload_s += time.perf_counter() - t_lut0

        blk_cyc_list = []
        n_tiles_l2 = m["l2_meta"]["n_tiles"]
        sdram_bytes_per_tile = m["l1_meta"]["total_bytes"] + m["l2_meta"]["total_bytes"]

        for ti in range(n_t_tiles):
            s0 = ti * T_TILE
            xq_tile = xq_full[s0 : s0 + T_TILE]
            yq_tile = yq_full[s0 : s0 + T_TILE]

            t_up0 = time.perf_counter()
            x_words = [int(s, 16) for s in pack_words_128bit_i8(xq_tile)]
            for idx, w in enumerate(x_words):
                dev.queue_write_128b(OP_COMMIT_ACT, idx, w)
            dev.flush()
            x_upload_s += time.perf_counter() - t_up0

            t_ex0 = time.perf_counter()
            c_l1 = run_sdram_stream_layer(
                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                dim_t=T_TILE, k_dim=d_m, n_dim=d_ff, layer_meta=m["l1_meta"],
                sdram_base_w128=m["l1_sdram_base_w128"], rd_lat=rd_lat, lut_mode=1,
                act_base=0, dst_base=128, dst_is_act=1,
            )
            c_l2 = run_sdram_stream_layer(
                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                dim_t=T_TILE, k_dim=d_ff, n_dim=d_m, layer_meta=m["l2_meta"],
                sdram_base_w128=m["l2_sdram_base_w128"], rd_lat=rd_lat, lut_mode=0,
                act_base=128, dst_base=0, dst_is_act=0,
            )
            hw_wait_exec_s += time.perf_counter() - t_ex0

            exp_csum = py_hw_checksum_stream(yq_tile, n_tiles_l2)
            hw_csum = int(c_l2["out_checksum"], 16)
            if hw_csum != exp_csum:
                csum_mismatches += 1
                raise AssertionError(f"PartA csum mismatch blk={m['block_idx']} tile={ti}: 0x{hw_csum:08x} != 0x{exp_csum:08x}")

            # Full byte-by-byte JTAG readback on first and last time tile of every block (72 tiles)
            if ti == 0 or ti == n_t_tiles - 1:
                y_gold = [int(s, 16) for s in pack_words_128bit_i8(yq_tile)]
                t_rb0 = time.perf_counter()
                y_hw = [dev.read_128b(0, i) for i in range(len(y_gold))]
                y_readback_s += time.perf_counter() - t_rb0
                total_readback_words += len(y_gold)
                total_readback_bytes += len(y_gold) * 16
                if y_hw != y_gold:
                    byte_mismatches += 1
                    raise AssertionError(f"PartA byte mismatch blk={m['block_idx']} tile={ti}")

            b_tot = c_l1["total_cycles"] + c_l2["total_cycles"]
            b_mac = c_l1["mac_cycles"] + c_l2["mac_cycles"]
            b_dma = c_l1["dma_cycles"] + c_l2["dma_cycles"]
            b_stall = c_l1["stall_cycles"] + c_l2["stall_cycles"]
            blk_cyc_list.append(b_tot)

            tot_cyc_all += b_tot
            mac_cyc_all += b_mac
            dma_cyc_all += b_dma
            stall_cyc_all += b_stall
            tot_macs_all += 2 * T_TILE * d_m * d_ff
            total_tiles_run += 1
            total_layer_launches += 2
            total_sdram_bytes_read += sdram_bytes_per_tile
            total_act_in_bytes += T_TILE * d_m
            total_act_out_bytes += T_TILE * d_m

            sa = stage_agg.setdefault(stage_id, {
                "stage": stage_id, "t_utt": t_real, "t_tiles_per_blk": n_t_tiles,
                "blocks": 6, "tiles_run": 0, "total_cyc": 0, "mac_cyc": 0, "stall_cyc": 0,
            })
            sa["tiles_run"] += 1
            sa["total_cyc"] += b_tot
            sa["mac_cyc"] += b_mac
            sa["stall_cyc"] += b_stall

        # T=500 exact shape scaling for comparison
        t_500 = int(sh500[pfx + "xq"][0])
        n_500 = (t_500 + T_TILE - 1) // T_TILE
        t_last = t_500 - (n_500 - 1) * T_TILE
        mean_tile_cyc = float(np.mean(blk_cyc_list))
        t500_macs_total += 2 * t_500 * d_m * d_ff
        t500_act_bytes_total += t_500 * d_m * 2
        t500_sdram_bytes_total += n_500 * sdram_bytes_per_tile
        t500_scaled_cyc_total += (n_500 - 1 + t_last / float(T_TILE)) * mean_tile_cyc
        t500_tiles_total += n_500

        if m["block_idx"] % 6 == 5 or m["block_idx"] == 0:
            print(
                f"  [PartA blk {m['block_idx']:02d}/35] {stage_id} T={t_real} ({n_t_tiles} tiles) "
                f"avg_cyc={mean_tile_cyc:.0f} -> 504-tile csum & byte-readback 0 LSB"
            )

    rb_rate_bps = total_readback_bytes / max(y_readback_s, 1e-6)
    up_rate_bps = total_act_in_bytes / max(x_upload_s, 1e-6)
    total_act_bytes_263 = total_act_in_bytes + total_act_out_bytes
    est_full_jtag_io_263_s = lut_upload_s + x_upload_s + (total_act_out_bytes / rb_rate_bps)

    return {
        "wav_file": DEFAULT_WAV.name,
        "wav_duration_s": round(audio_dur_s, 4),
        "fbank_frames": int(len(feats)),
        "t_utt_by_stage_t263": t_utt_by_stage,
        "measured_full_sequence_t263": {
            "blocks": len(manifest),
            "total_block_tiles_run": total_tiles_run,
            "total_layer_launches": total_layer_launches,
            "csum_mismatches_all_504": csum_mismatches,
            "byte_readback_tiles_checked": 2 * len(manifest),
            "byte_readback_words_checked": total_readback_words,
            "byte_mismatches": byte_mismatches,
            "ffn_macs_executed": tot_macs_all,
            "total_cycles": tot_cyc_all,
            "mac_cycles": mac_cyc_all,
            "dma_cycles": dma_cyc_all,
            "stall_cycles": stall_cyc_all,
            "mac_lane_utilization_pct": round(100.0 * mac_cyc_all / tot_cyc_all, 2),
            "stall_pct": round(100.0 * stall_cyc_all / tot_cyc_all, 2),
            "sdram_bytes_read": total_sdram_bytes_read,
            "sdram_MiB_read": round(total_sdram_bytes_read / (1024.0 * 1024.0), 2),
            "host_act_in_bytes": total_act_in_bytes,
            "host_act_out_bytes": total_act_out_bytes,
            "host_act_total_bytes": total_act_bytes_263,
            "host_act_total_MiB": round(total_act_bytes_263 / (1024.0 * 1024.0), 4),
            "measured_jtag_write_Bps": round(up_rate_bps, 1),
            "measured_jtag_read_Bps": round(rb_rate_bps, 1),
            "measured_wall_x_upload_s": round(x_upload_s, 3),
            "measured_wall_lut_upload_s": round(lut_upload_s, 3),
            "measured_wall_72tile_readback_s": round(y_readback_s, 3),
            "est_full_bidir_jtag_io_s": round(est_full_jtag_io_263_s, 2),
        },
        "t500_normalized_projection": {
            "ffn_macs_total": t500_macs_total,
            "total_tiles_n": t500_tiles_total,
            "sdram_bytes_read": t500_sdram_bytes_total,
            "sdram_MiB_read": round(t500_sdram_bytes_total / (1024.0 * 1024.0), 2),
            "sdram_weight_reuse_factor": round(t500_sdram_bytes_total / 6832128.0, 2),
            "host_act_total_bytes": t500_act_bytes_total,
            "host_act_total_KiB": round(t500_act_bytes_total / 1024.0, 1),
            "host_act_total_MiB": round(t500_act_bytes_total / (1024.0 * 1024.0), 4),
            "scaled_cycles": round(t500_scaled_cyc_total),
        },
        "stage_breakdown_t263": list(stage_agg.values()),
    }


def run_part_b_50_utts_and_stress(
    dev: UsbBlasterJtag,
    sess: ort.InferenceSession,
    manifest: list[dict],
    rd_lat: int,
) -> dict:
    """Evaluate 50 diverse VIVOS utterances + 4 adversarial corner stress tiles per block on live FPGA."""
    wavs = pick_stratified_vivos_wavs(VIVOS_TEST_WAVES, target_count=50)
    speakers = sorted({w.parent.name for w in wavs})
    print(f"[PartB] Selected {len(wavs)} VIVOS test utterances across {len(speakers)} speakers.")

    # Full-sequence Python + ONNX statistics across all 50 utterances x 36 blocks
    l1_global_stats: dict[str, int] = {}
    l2_global_stats: dict[str, int] = {}
    x_global_rails = {"elements": 0, "rail_pos_127": 0, "rail_neg_128": 0}
    hpost_global_rails = {"elements": 0, "rail_pos_127": 0, "rail_neg_128": 0}
    peak_tiles_per_block: list[list[tuple[str, np.ndarray, np.ndarray]]] = [[] for _ in range(len(manifest))]

    utt_metas = []
    t_prep0 = time.perf_counter()
    for wi, wpath in enumerate(wavs):
        audio, sr = sf.read(str(wpath), dtype="float32")
        feats = compute_fbank(audio, sr)
        outs = sess.run(None, {"x": feats[None], "x_lens": np.array([len(feats)], np.int64)})
        out_map = {o.name: outs[i] for i, o in enumerate(sess.get_outputs())}
        utt_metas.append({"wav": wpath.name, "speaker": wpath.parent.name, "fbank_frames": int(len(feats)), "dur_s": round(len(audio) / float(sr), 3)})

        for bi, m in enumerate(manifest):
            pfx = f"_fpga_{m['tag']}_"
            d_m = m["d_model"]
            xq_full = np.rint(out_map[pfx + "xq"]).astype(np.int8).reshape(-1, d_m)
            x_global_rails["elements"] += int(xq_full.size)
            x_global_rails["rail_pos_127"] += int(np.sum(xq_full == 127))
            x_global_rails["rail_neg_128"] += int(np.sum(xq_full == -128))

            _, hp_full, yq_full, l1_st, l2_st, row_score, _ = eval_block_python(m, xq_full)
            _accum_stats(l1_global_stats, l1_st)
            _accum_stats(l2_global_stats, l2_st)
            hpost_global_rails["elements"] += int(hp_full.size)
            hpost_global_rails["rail_pos_127"] += int(np.sum(hp_full == 127))
            hpost_global_rails["rail_neg_128"] += int(np.sum(hp_full == -128))

            t_len = xq_full.shape[0]
            if t_len >= T_TILE:
                # Pick T_TILE window maximizing combined clamps + exact rounding ties
                win_scores = [int(np.sum(row_score[s : s + T_TILE])) for s in range(0, t_len - T_TILE + 1)]
                best_s = int(np.argmax(win_scores))
                xq_win = xq_full[best_s : best_s + T_TILE].copy()
                yq_win = yq_full[best_s : best_s + T_TILE].copy()
            else:
                pad_rows = T_TILE - t_len
                xq_win = np.vstack([xq_full, np.zeros((pad_rows, d_m), dtype=np.int8)])
                _, _, yq_win, _, _, _, _ = eval_block_python(m, xq_win)
            peak_tiles_per_block[bi].append((wpath.name, xq_win, yq_win))

    prep_s = time.perf_counter() - t_prep0
    print(
        f"[PartB] 50-utt ONNX+Python full-seq scan ({prep_s:.1f}s): "
        f"L1 elems={l1_global_stats['elements']:,}, strict_clamp=({l1_global_stats['strict_clamp_neg']}+{l1_global_stats['strict_clamp_pos']}), "
        f"0.5LSB_ties={l1_global_stats['exact_half_ties']}; "
        f"L2 elems={l2_global_stats['elements']:,}, strict_clamp=({l2_global_stats['strict_clamp_neg']}+{l2_global_stats['strict_clamp_pos']}), "
        f"0.5LSB_ties={l2_global_stats['exact_half_ties']}"
    )

    # Construct 4 adversarial corner-case stress tiles per block & gather their integer stats
    rng = np.random.default_rng(20261118)
    stress_tiles_per_block: list[list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = []
    stress_l1_stats: dict[str, dict] = {}
    stress_l2_stats: dict[str, dict] = {}
    for m in manifest:
        st_map = construct_corner_stress_tiles(m, rng)
        blk_st = []
        for name, xq_s in st_map.items():
            _, hp_s, yq_s, l1_st, l2_st, _, _ = eval_block_python(m, xq_s)
            _accum_stats(stress_l1_stats.setdefault(name, {}), l1_st)
            _accum_stats(stress_l2_stats.setdefault(name, {}), l2_st)
            blk_st.append((name, xq_s, hp_s, yq_s))
        stress_tiles_per_block.append(blk_st)

    # Hardware execution across all 36 blocks: outer loop over block (write LUT once per block)
    speech_tiles_run = 0
    speech_csum_mism = 0
    speech_tot_cyc = 0
    speech_mac_cyc = 0
    speech_stall_cyc = 0

    stress_tiles_run = 0
    stress_csum_mism = 0
    stress_y_byte_mism = 0
    stress_hpost_byte_mism = 0
    stress_y_bytes_checked = 0
    stress_hpost_bytes_checked = 0
    stress_per_type_ok: dict[str, int] = {}

    t_hw0 = time.perf_counter()
    for bi, m in enumerate(manifest):
        d_m, d_ff = m["d_model"], m["d_ff"]
        n_tiles_l2 = m["l2_meta"]["n_tiles"]
        write_lut_256(dev, m["lut_i8"], lut_bank=0)

        # 50 real-speech peak tiles for this block
        for _, xq_win, yq_win in peak_tiles_per_block[bi]:
            for idx, w in enumerate(int(s, 16) for s in pack_words_128bit_i8(xq_win)):
                dev.queue_write_128b(OP_COMMIT_ACT, idx, w)
            dev.flush()
            c_l1 = run_layer_fast(dev, T_TILE, d_m, d_ff, m["l1_meta"], m["l1_sdram_base_w128"], rd_lat, 1, 0, 128, 1)
            c_l2 = run_layer_fast(dev, T_TILE, d_ff, d_m, m["l2_meta"], m["l2_sdram_base_w128"], rd_lat, 0, 128, 0, 0)
            exp_csum = py_hw_checksum_stream(yq_win, n_tiles_l2)
            if int(c_l2["out_checksum"], 16) != exp_csum:
                speech_csum_mism += 1
                raise AssertionError(f"PartB speech csum mismatch blk={bi}")
            speech_tiles_run += 1
            speech_tot_cyc += c_l1["total_cycles"] + c_l2["total_cycles"]
            speech_mac_cyc += c_l1["mac_cycles"] + c_l2["mac_cycles"]
            speech_stall_cyc += c_l1["stall_cycles"] + c_l2["stall_cycles"]

        # 4 adversarial corner-case tiles for this block (FULL byte-by-byte y_out on 100%, + h_post on stress_max_rail_saturation)
        for s_name, xq_s, hp_s, yq_s in stress_tiles_per_block[bi]:
            for idx, w in enumerate(int(s, 16) for s in pack_words_128bit_i8(xq_s)):
                dev.queue_write_128b(OP_COMMIT_ACT, idx, w)
            dev.flush()
            c_l1 = run_layer_fast(dev, T_TILE, d_m, d_ff, m["l1_meta"], m["l1_sdram_base_w128"], rd_lat, 1, 0, 128, 1)
            c_l2 = run_layer_fast(dev, T_TILE, d_ff, d_m, m["l2_meta"], m["l2_sdram_base_w128"], rd_lat, 0, 128, 0, 0)
            exp_csum = py_hw_checksum_stream(yq_s, n_tiles_l2)
            if int(c_l2["out_checksum"], 16) != exp_csum:
                stress_csum_mism += 1
                raise AssertionError(f"PartB stress csum mismatch blk={bi} type={s_name}")

            y_gold = [int(s, 16) for s in pack_words_128bit_i8(yq_s)]
            y_hw = [dev.read_128b(0, i) for i in range(len(y_gold))]
            stress_y_bytes_checked += len(y_gold) * 16
            if y_hw != y_gold:
                stress_y_byte_mism += 1
                raise AssertionError(f"PartB stress byte mismatch blk={bi} type={s_name}")

            if s_name == "stress_max_rail_saturation":
                hp_gold = [int(s, 16) for s in pack_words_128bit_i8(hp_s)]
                hp_hw = [dev.read_128b(1, 128 + i) for i in range(len(hp_gold))]
                stress_hpost_bytes_checked += len(hp_gold) * 16
                if hp_hw != hp_gold:
                    stress_hpost_byte_mism += 1
                    raise AssertionError(f"PartB stress h_post mismatch blk={bi}")

            stress_tiles_run += 1
            stress_per_type_ok[s_name] = stress_per_type_ok.get(s_name, 0) + 1

        if bi % 6 == 5 or bi == 0:
            print(
                f"  [PartB blk {bi:02d}/35] 50 real + 4 stress = 54 tiles verified "
                f"(cumulative: {speech_tiles_run + stress_tiles_run}/1944, wall={time.perf_counter()-t_hw0:.1f}s)"
            )

    hw_wall_s = time.perf_counter() - t_hw0
    return {
        "utterances_count": len(wavs),
        "speakers_covered": len(speakers),
        "utterance_list": utt_metas,
        "dataset_50utt_fullseq_integer_stats": {
            "x_q_rails": x_global_rails,
            "l1_pre_lut_requant": l1_global_stats,
            "h_post_lut_rails": hpost_global_rails,
            "l2_out_requant": l2_global_stats,
        },
        "fpga_50utt_peak_windows": {
            "blocks_per_utt": len(manifest),
            "tiles_verified_on_fpga": speech_tiles_run,
            "layer_launches": speech_tiles_run * 2,
            "checksum_mismatches": speech_csum_mism,
            "total_cycles": speech_tot_cyc,
            "mac_cycles": speech_mac_cyc,
            "stall_cycles": speech_stall_cyc,
            "mac_lane_utilization_pct": round(100.0 * speech_mac_cyc / speech_tot_cyc, 2),
            "stall_pct": round(100.0 * speech_stall_cyc / speech_tot_cyc, 2),
        },
        "fpga_adversarial_corner_stress": {
            "tiles_verified_on_fpga": stress_tiles_run,
            "layer_launches": stress_tiles_run * 2,
            "checksum_mismatches": stress_csum_mism,
            "y_out_byte_readback_bytes": stress_y_bytes_checked,
            "y_out_byte_mismatches": stress_y_byte_mism,
            "h_post_byte_readback_bytes": stress_hpost_bytes_checked,
            "h_post_byte_mismatches": stress_hpost_byte_mism,
            "tiles_ok_per_stress_type": stress_per_type_ok,
            "l1_stats_per_stress_type": stress_l1_stats,
            "l2_stats_per_stress_type": stress_l2_stats,
        },
        "combined_hardware_totals": {
            "total_block_tiles_verified": speech_tiles_run + stress_tiles_run,
            "total_layer_launches": (speech_tiles_run + stress_tiles_run) * 2,
            "zero_lsb_mismatches": speech_csum_mism + stress_csum_mism + stress_y_byte_mism + stress_hpost_byte_mism,
            "wall_time_s": round(hw_wall_s, 2),
        },
    }


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    all_words, manifest, img_meta = pack_all_36_ffn_image()
    sess = build_probe_session(manifest)

    with UsbBlasterJtag() as dev:
        cap = dev.read_resp(0x2005)
        assert (cap >> 32) & 1 == 1, "SDRAM not ready!"
        eng_mhz, sd_mhz, _, _ = measure_pll_clocks(dev)
        already_ok, up_s = verify_or_upload_sdram(dev, all_words, manifest)
        print(f"[G5b-Multi] FPGA eng={eng_mhz} MHz, sdram={sd_mhz} MHz | SDRAM already resident={already_ok} (upload={up_s:.2f}s)")

        part_a = run_part_a_full_sequence(dev, sess, manifest, rd_lat=4)
        p_t263 = part_a["measured_full_sequence_t263"]
        p_t500 = part_a["t500_normalized_projection"]
        t263_ms = 1e3 * p_t263["total_cycles"] / (eng_mhz * 1e6)
        t500_ms = 1e3 * p_t500["scaled_cycles"] / (eng_mhz * 1e6)
        p_t263["measured_hw_compute_ms"] = round(t263_ms, 2)
        p_t263["measured_ffn_rtf_vs_wav"] = round((t263_ms / 1e3) / part_a["wav_duration_s"], 5)
        p_t263["measured_ffn_rtf_vs_263frames"] = round((t263_ms / 1e3) / (part_a["fbank_frames"] * 0.01), 5)
        p_t500["scaled_hw_compute_ms"] = round(t500_ms, 2)
        p_t500["scaled_ffn_rtf_vs_5s"] = round(t500_ms / 5000.0, 5)
        print(
            f"[PartA DONE] 504 tiles on T=263 ({part_a['wav_duration_s']}s): "
            f"{t263_ms:.2f} ms (measured RTF_FFN={p_t263['measured_ffn_rtf_vs_wav']}), "
            f"act_IO={p_t263['host_act_total_MiB']} MiB | T=500 scaled={t500_ms:.2f} ms "
            f"(scaled RTF_FFN={p_t500['scaled_ffn_rtf_vs_5s']}), act_IO={p_t500['host_act_total_MiB']} MiB"
        )

        part_b = run_part_b_50_utts_and_stress(dev, sess, manifest, rd_lat=4)
        print(
            f"[PartB DONE] Verified {part_b['combined_hardware_totals']['total_block_tiles_verified']} block-tiles "
            f"({part_b['combined_hardware_totals']['total_layer_launches']} layers) on hardware with "
            f"{part_b['combined_hardware_totals']['zero_lsb_mismatches']} mismatches!"
        )

    out = {
        "gate": "G5b_Extended",
        "device": "EP4CE10E22C8N",
        "sdram_chip": "HY57V641620FTP-H (8 MiB, 16-bit)",
        "eng_mhz": eng_mhz,
        "sdram_mhz": sd_mhz,
        "rd_lat": 4,
        "sdram_spotcheck_reused_resident_image": already_ok,
        "part_a_full_sequence_measurement": part_a,
        "part_b_50_utterances_and_corner_stress": part_b,
    }
    out_path = BUILD_DIR / "g5b_multisent_fullseq_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
