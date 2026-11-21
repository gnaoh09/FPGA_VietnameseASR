#!/usr/bin/env python3
"""Shared helpers for packing W4A8 FFN layers and executing them from SDRAM via Ping-Pong DMA."""

from __future__ import annotations

import numpy as np

from fpga_jtag_runner import OP_START, OP_WRITE_LUT, OP_WRITE_REG, UsbBlasterJtag
from gen_golden_vectors import pack_params_64bit, pack_words_128bit_w4


def pack_cfg4(stream_en: int, ntiles: int, kchunks: int, rd_lat: int, w_words: int) -> int:
    return (
        (stream_en & 0x1)
        | ((ntiles & 0xFF) << 1)
        | ((kchunks & 0xF) << 9)
        | ((rd_lat & 0x7) << 13)
        | ((w_words & 0x1FF) << 16)
    )


def choose_k_chunks(k_dim: int, max_w_words: int = 256) -> int:
    """Pick smallest divisor k_chunks of k_dim so (k_dim / k_chunks) % 32 == 0 and w_words <= max_w_words.
    Each chunk has 16 rows * (k_chunk / 32) = k_chunk / 2 128-bit words in W4 mode, plus even k16.
    """
    assert k_dim % 32 == 0, f"k_dim={k_dim} must be multiple of 32 for W4"
    for kc in range(1, 16):
        if k_dim % kc != 0:
            continue
        k_chunk = k_dim // kc
        if (k_chunk % 32 == 0) and (k_chunk // 2 <= max_w_words):
            return kc
    raise ValueError(f"Cannot partition K={k_dim} into <=15 chunks with w_words<={max_w_words}")


def pack_layer_w4_sdram_words(
    w_q: np.ndarray,
    b_q: np.ndarray,
    m16: np.ndarray,
    sh: np.ndarray,
) -> tuple[list[int], dict]:
    """Pack one layer W_q[N, K] (int4 in int8) + params[N] into uniform SDRAM DMA packets.
    Packet for each (nt, kc):
      - 8 x 128-bit words = 16 x 64-bit {bias_i32, m16, shift} for output rows 16*nt..16*nt+15
      - w_words x 128-bit words = 16 rows x (k_chunk / 32) W4 words
    """
    n_dim, k_dim = w_q.shape
    assert n_dim % 16 == 0, f"N={n_dim} must be multiple of 16"
    n_tiles = n_dim // 16
    k_chunks = choose_k_chunks(k_dim)
    k_chunk = k_dim // k_chunks
    w_words_per_chunk = k_chunk // 2

    words: list[int] = []
    for nt in range(n_tiles):
        r0, r1 = nt * 16, (nt + 1) * 16
        param_lines = pack_params_64bit(b_q[r0:r1], m16[r0:r1], sh[r0:r1])
        param_ints = [int(s, 16) for s in param_lines]
        param_128s = [param_ints[2 * i] | (param_ints[2 * i + 1] << 64) for i in range(8)]
        for kc in range(k_chunks):
            c0, c1 = kc * k_chunk, (kc + 1) * k_chunk
            wgt_lines = pack_words_128bit_w4(w_q[r0:r1, c0:c1])
            assert len(wgt_lines) == w_words_per_chunk
            words.extend(param_128s)
            words.extend(int(s, 16) for s in wgt_lines)

    meta = {
        "N": int(n_dim),
        "K": int(k_dim),
        "n_tiles": int(n_tiles),
        "k_chunks": int(k_chunks),
        "k_chunk": int(k_chunk),
        "w_words_per_chunk": int(w_words_per_chunk),
        "pkt_words": int(8 + w_words_per_chunk),
        "total_words_128b": len(words),
        "total_bytes": len(words) * 16,
        "wgt_bytes": (n_dim * k_dim) // 2,
        "param_bytes": n_tiles * k_chunks * 8 * 16,
    }
    return words, meta


def write_lut_256(dev: UsbBlasterJtag, lut_i8: np.ndarray, lut_bank: int = 0) -> None:
    u8 = lut_i8.view(np.uint8)
    base = 256 if lut_bank else 0
    for idx in range(256):
        dev.queue_cmd(OP_WRITE_LUT, base + idx, int(u8[idx]))
    dev.flush()


def run_sdram_stream_layer(
    dev: UsbBlasterJtag,
    pack_cfg0_fn,
    pack_cfg3_fn,
    pack_cfg1_fn,
    pack_cfg2_fn,
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
    """Execute an entire N_tiles x K_chunks W4A8 layer using hardware Ping-Pong SDRAM DMA."""
    n_tiles = layer_meta["n_tiles"]
    k_chunks = layer_meta["k_chunks"]
    k_chunk = layer_meta["k_chunk"]
    w_words = layer_meta["w_words_per_chunk"]

    dev.queue_cmd(OP_WRITE_REG, 0, pack_cfg0_fn(0, 1, lut_mode, dim_t, k_chunk, 16, acc_mode=0))
    dev.queue_cmd(OP_WRITE_REG, 1, pack_cfg1_fn(16, act_base & 0xFF, 0, dst_base & 0xFF))
    dev.queue_cmd(OP_WRITE_REG, 2, pack_cfg2_fn(0, 0, 0, dst_base & 0xFF))
    dev.queue_cmd(
        OP_WRITE_REG,
        3,
        pack_cfg3_fn(
            0,
            k_chunk,
            act_base,
            dst_base,
            dst_stride=n_dim // 16,
            stress_loop=0,
            act_stride=k_dim // 16,
            dst_is_act=dst_is_act,
        ),
    )
    dev.queue_cmd(OP_WRITE_REG, 4, pack_cfg4(1, n_tiles, k_chunks, rd_lat, w_words))
    dev.queue_cmd(OP_WRITE_REG, 5, sdram_base_w128 & 0x7FFFF)
    dev.queue_cmd(OP_START, 0, 0)
    dev.flush()

    cnt = dev.wait_done(timeout_s=5.0)
    r9 = dev.read_resp(0x2009)
    r11 = dev.read_resp(0x200B)
    cnt["stall_cycles"] = (r9 >> 32) & 0xFFFFFFFF
    cnt["dma_cycles"] = (r11 >> 32) & 0xFFFFFFFF
    return cnt
