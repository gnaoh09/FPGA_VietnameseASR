#!/usr/bin/env python3
"""Q2 Priority #1: Full 2,448 / 2,448 Block-Tile Byte-by-Byte y_out Verification
on Live Cyclone IV EP4CE10E22C8N + 100 MHz SDR SDRAM (HY57V641620FTP-H).

Closes the checksum-only gap across all 3 campaigns:
  - Campaign 1 (Full-sequence T=263, VIVOSDEV02_R106.wav): 504 / 504 tiles y_out read back byte-by-byte
  - Campaign 2 (50 speaker-stratified VIVOS utterances):  1,800 / 1,800 peak tiles y_out read back byte-by-byte
  - Campaign 3 (4 adversarial corner stress vectors):   144 / 144 tiles y_out + 144 / 144 tiles h_post read back byte-by-byte

Reports exact bytes checked, byte mismatches, max_abs_diff, and keeps h_post SwooshL
observation scope cleanly separated from y_out coverage.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fpga_bitexact_30m_onnx import compute_fbank
from fpga_jtag_runner import OP_COMMIT_ACT, UsbBlasterJtag
from gen_golden_vectors import pack_words_128bit_i8
from run_g4b_sdram_experiment import measure_pll_clocks
from run_g5b_30m_resident_sdram import DEFAULT_WAV, pack_all_36_ffn_image
from run_g5b_multisent_fullseq import (
    VIVOS_TEST_WAVES,
    build_probe_session,
    construct_corner_stress_tiles,
    eval_block_python,
    pick_stratified_vivos_wavs,
    py_hw_checksum_stream,
    run_layer_fast,
    verify_or_upload_sdram,
)
from sdram_helpers import write_lut_256

import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
T_TILE = 4
RD_LAT = 4


def unpack_words_128bit_i8(words: list[int], rows: int, cols: int) -> np.ndarray:
    """Unpack 128-bit hardware words back into int8 [rows, cols] array."""
    raw = b"".join(w.to_bytes(16, "little") for w in words)
    return np.frombuffer(raw, dtype=np.int8).reshape(rows, cols).copy()


def read_and_compare_tensor(
    dev: UsbBlasterJtag,
    mem_id: int,
    base_word: int,
    golden_q: np.ndarray,
) -> tuple[int, int, int, float]:
    """Read full tensor word-by-word via JTAG, returning (bytes, byte_mism, max_abs_diff, elapsed_s)."""
    gold_words = [int(s, 16) for s in pack_words_128bit_i8(golden_q)]
    t0 = time.perf_counter()
    hw_words = [dev.read_128b(mem_id, base_word + i) for i in range(len(gold_words))]
    elapsed = time.perf_counter() - t0
    hw_q = unpack_words_128bit_i8(hw_words, golden_q.shape[0], golden_q.shape[1])
    hw_u8 = hw_q.view(np.uint8)
    g_u8 = golden_q.view(np.uint8)
    byte_mism = int(np.sum(hw_u8 != g_u8))
    max_abs = int(np.max(np.abs(hw_q.astype(np.int16) - golden_q.astype(np.int16))))
    return len(gold_words) * 16, byte_mism, max_abs, elapsed


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    all_words, manifest, _ = pack_all_36_ffn_image()
    sess = build_probe_session(manifest)

    # Precompute Campaign 1 (T=263 full sequence)
    audio_a, sr_a = sf.read(str(DEFAULT_WAV), dtype="float32")
    feats_a = compute_fbank(audio_a, sr_a)
    outs_a = sess.run(None, {"x": feats_a[None], "x_lens": np.array([len(feats_a)], np.int64)})
    out_map_a = {o.name: outs_a[i] for i, o in enumerate(sess.get_outputs())}
    part_a_tiles: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for m in manifest:
        pfx = f"_fpga_{m['tag']}_"
        xq_full = np.rint(out_map_a[pfx + "xq"]).astype(np.int8).reshape(-1, m["d_model"])
        _, _, yq_full, _, _, _, _ = eval_block_python(m, xq_full)
        n_t = xq_full.shape[0] // T_TILE
        part_a_tiles.append([
            (xq_full[ti * T_TILE : (ti + 1) * T_TILE], yq_full[ti * T_TILE : (ti + 1) * T_TILE])
            for ti in range(n_t)
        ])

    # Precompute Campaign 2 (50 diverse VIVOS utterances, peak window per block)
    wavs_b = pick_stratified_vivos_wavs(VIVOS_TEST_WAVES, target_count=50)
    peak_tiles_per_block: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(len(manifest))]
    for wpath in wavs_b:
        aud, sr = sf.read(str(wpath), dtype="float32")
        fb = compute_fbank(aud, sr)
        outs = sess.run(None, {"x": fb[None], "x_lens": np.array([len(fb)], np.int64)})
        omap = {o.name: outs[i] for i, o in enumerate(sess.get_outputs())}
        for bi, m in enumerate(manifest):
            pfx = f"_fpga_{m['tag']}_"
            d_m = m["d_model"]
            xq_full = np.rint(omap[pfx + "xq"]).astype(np.int8).reshape(-1, d_m)
            _, _, yq_full, _, _, row_score, _ = eval_block_python(m, xq_full)
            t_len = xq_full.shape[0]
            if t_len >= T_TILE:
                win_scores = [int(np.sum(row_score[s : s + T_TILE])) for s in range(0, t_len - T_TILE + 1)]
                best_s = int(np.argmax(win_scores))
                peak_tiles_per_block[bi].append((xq_full[best_s : best_s + T_TILE].copy(), yq_full[best_s : best_s + T_TILE].copy()))
            else:
                pad = T_TILE - t_len
                xw = np.vstack([xq_full, np.zeros((pad, d_m), dtype=np.int8)])
                _, _, yw, _, _, _, _ = eval_block_python(m, xw)
                peak_tiles_per_block[bi].append((xw, yw))

    # Precompute Campaign 3 (4 adversarial stress vectors x 36 blocks = 144 tiles)
    rng = np.random.default_rng(20261118)
    stress_per_block: list[list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = []
    for m in manifest:
        st_map = construct_corner_stress_tiles(m, rng)
        blk_st = []
        for s_name, xq_s in st_map.items():
            _, hp_s, yq_s, _, _, _, _ = eval_block_python(m, xq_s)
            blk_st.append((s_name, xq_s, hp_s, yq_s))
        stress_per_block.append(blk_st)

    stats = {
        "full_seq_t263": {"tiles": 0, "y_out_bytes": 0, "y_out_byte_mism": 0, "csum_mism": 0, "max_abs_diff": 0, "read_s": 0.0},
        "vivos_50utt": {"tiles": 0, "y_out_bytes": 0, "y_out_byte_mism": 0, "csum_mism": 0, "max_abs_diff": 0, "read_s": 0.0},
        "stress_4x36": {"tiles": 0, "y_out_bytes": 0, "y_out_byte_mism": 0, "h_post_bytes": 0, "h_post_byte_mism": 0, "csum_mism": 0, "max_abs_diff": 0, "read_s": 0.0},
    }

    with UsbBlasterJtag() as dev:
        cap = dev.read_resp(0x2005)
        assert (cap >> 32) & 1 == 1, "SDRAM not ready!"
        eng_mhz, sd_mhz, _, _ = measure_pll_clocks(dev)
        already_ok, up_s = verify_or_upload_sdram(dev, all_words, manifest)
        print(f"[Full-2448] eng={eng_mhz} MHz, sdram={sd_mhz} MHz | resident_reused={already_ok} (upload={up_s:.1f}s)")

        t0_all = time.perf_counter()
        for bi, m in enumerate(manifest):
            d_m, d_ff = m["d_model"], m["d_ff"]
            n_tiles_l2 = m["l2_meta"]["n_tiles"]
            write_lut_256(dev, m["lut_i8"], lut_bank=0)

            def run_and_verify_tile(xq_t: np.ndarray, yq_t: np.ndarray, hp_t: np.ndarray | None, bucket: dict) -> None:
                for idx_w, w in enumerate(int(s, 16) for s in pack_words_128bit_i8(xq_t)):
                    dev.queue_write_128b(OP_COMMIT_ACT, idx_w, w)
                dev.flush()
                _ = run_layer_fast(dev, T_TILE, d_m, d_ff, m["l1_meta"], m["l1_sdram_base_w128"], RD_LAT, 1, 0, 128, 1)
                c_l2 = run_layer_fast(dev, T_TILE, d_ff, d_m, m["l2_meta"], m["l2_sdram_base_w128"], RD_LAT, 0, 128, 0, 0)
                if int(c_l2["out_checksum"], 16) != py_hw_checksum_stream(yq_t, n_tiles_l2):
                    bucket["csum_mism"] += 1
                    raise AssertionError(f"Checksum mismatch blk={bi}")
                nb, bm, mad, dt = read_and_compare_tensor(dev, 0, 0, yq_t)
                bucket["tiles"] += 1
                bucket["y_out_bytes"] += nb
                bucket["y_out_byte_mism"] += bm
                bucket["max_abs_diff"] = max(bucket["max_abs_diff"], mad)
                bucket["read_s"] += dt
                assert bm == 0 and mad == 0, f"y_out byte mismatch blk={bi}: bm={bm}, mad={mad}"

                if hp_t is not None:
                    nh, bmh, madh, dth = read_and_compare_tensor(dev, 1, 128, hp_t)
                    bucket["h_post_bytes"] += nh
                    bucket["h_post_byte_mism"] += bmh
                    bucket["max_abs_diff"] = max(bucket["max_abs_diff"], madh)
                    bucket["read_s"] += dth
                    assert bmh == 0 and madh == 0, f"h_post byte mismatch blk={bi}: bmh={bmh}, madh={madh}"

            for xq_t, yq_t in part_a_tiles[bi]:
                run_and_verify_tile(xq_t, yq_t, None, stats["full_seq_t263"])
            for xq_t, yq_t in peak_tiles_per_block[bi]:
                run_and_verify_tile(xq_t, yq_t, None, stats["vivos_50utt"])
            for _s_name, xq_s, hp_s, yq_s in stress_per_block[bi]:
                run_and_verify_tile(xq_s, yq_s, hp_s, stats["stress_4x36"])

            tot_done = sum(s["tiles"] for s in stats.values())
            tot_yb = sum(s["y_out_bytes"] for s in stats.values())
            print(
                f"  [blk {bi:02d}/35] tiles={tot_done}/2448 | y_out={tot_yb:,} B (0 err) | wall={time.perf_counter()-t0_all:.1f}s",
                flush=True,
            )

    tot_tiles = sum(s["tiles"] for s in stats.values())
    tot_y_bytes = sum(s["y_out_bytes"] for s in stats.values())
    tot_y_mism = sum(s["y_out_byte_mism"] for s in stats.values())
    tot_hp_bytes = stats["stress_4x36"]["h_post_bytes"]
    tot_hp_mism = stats["stress_4x36"]["h_post_byte_mism"]
    tot_csum_mism = sum(s["csum_mism"] for s in stats.values())
    global_mad = max(s["max_abs_diff"] for s in stats.values())
    tot_read_s = sum(s["read_s"] for s in stats.values())

    assert tot_tiles == 2448
    out = {
        "experiment": "Q2_Priority_1_Full_2448_Byte_Verification",
        "device": "EP4CE10E22C8N",
        "eng_mhz": eng_mhz,
        "sdram_mhz": sd_mhz,
        "sdram_image_reused": already_ok,
        "sdram_upload_s": round(up_s, 2),
        "wall_time_s": round(time.perf_counter() - t0_all, 2),
        "jtag_readback_wall_s": round(tot_read_s, 2),
        "effective_jtag_read_Bps": round((tot_y_bytes + tot_hp_bytes) / max(tot_read_s, 1e-6), 1),
        "campaigns": stats,
        "totals": {
            "y_out_tiles_byte_verified": tot_tiles,
            "y_out_tiles_expected": 2448,
            "y_out_total_bytes_checked": tot_y_bytes,
            "y_out_byte_mismatches": tot_y_mism,
            "h_post_tiles_byte_verified": stats["stress_4x36"]["tiles"],
            "h_post_total_bytes_checked": tot_hp_bytes,
            "h_post_byte_mismatches": tot_hp_mism,
            "total_all_tensor_bytes_checked": tot_y_bytes + tot_hp_bytes,
            "csum_mismatches": tot_csum_mism,
            "global_max_abs_diff_lsb": global_mad,
        },
    }
    out_path = BUILD_DIR / "full_2448_byte_verify_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[FULL-2448 PASS]", json.dumps(out["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
