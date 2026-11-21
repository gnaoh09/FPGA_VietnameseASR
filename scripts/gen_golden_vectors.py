#!/usr/bin/env python3
"""Generate deterministic bit-accurate golden test vectors (.hex + .npz + metadata.json)
shared identically by Python verification, Icarus Verilog RTL simulation, and FPGA JTAG tests.
Workloads:
  1. golden_vectors/stage0_layer0_ffn1_W8A8
  2. golden_vectors/stage0_layer0_ffn1_W4A8
  3. golden_vectors/joiner_W8A8
"""

import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT_DIR / "golden_vectors"

SEED = 20261117
T_FRAMES = 16
K_DIM = 32
H_DIM = 64
V_DIM = 32


def swoosh_l_fp32(x: np.ndarray) -> np.ndarray:
    z = x - 1.0
    softplus = np.where(z > 20.0, z, np.log1p(np.exp(np.clip(z, -40.0, 20.0))))
    return softplus - 0.08 * x - 0.313261687


def make_dyadic_m16(real_scale: float) -> tuple[int, int]:
    """Return (m16, shift) with m16 in [1, 32767] and shift in [1, 30]."""
    if real_scale <= 0.0:
        return 1, 15
    exp = math.floor(math.log2(real_scale))
    shift = max(1, min(30, 14 - exp))
    m16 = int(round(real_scale * (1 << shift)))
    while m16 > 32767 and shift > 1:
        shift -= 1
        m16 = int(round(real_scale * (1 << shift)))
    return max(1, min(32767, m16)), shift


def dyadic_requantize_round_half_up(acc_i32: np.ndarray, m16: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Bit-accurate match to Verilog signed [47:0] (acc * m16 + (1 << (shift-1))) >>> shift, clamp [-128, 127]."""
    out = np.empty_like(acc_i32, dtype=np.int8)
    rows, cols = acc_i32.shape
    for c in range(cols):
        m = int(m16[c])
        s = int(shift[c])
        half = 1 << (s - 1) if s > 0 else 0
        for r in range(rows):
            prod = int(acc_i32[r, c]) * m
            val = (prod + half) >> s if s > 0 else prod
            if val > 127:
                val = 127
            elif val < -128:
                val = -128
            out[r, c] = np.int8(val)
    return out


def build_lut_256(fn, scale_in: float, scale_out: float) -> np.ndarray:
    """256-entry int8 LUT indexed directly by uint8 view of two's-complement int8."""
    lut = np.zeros(256, dtype=np.int8)
    for u in range(256):
        s_val = u - 256 if u >= 128 else u
        x_f = float(s_val) * scale_in
        y_f = float(fn(np.array([x_f], dtype=np.float64))[0])
        q = int(round(y_f / scale_out))
        q = max(-128, min(127, q))
        lut[u] = np.int8(q)
    return lut


def apply_lut_256(q_in: np.ndarray, lut: np.ndarray) -> np.ndarray:
    u_idx = q_in.view(np.uint8)
    return lut[u_idx]


def sqnr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    sig = float(np.mean(ref.astype(np.float64) ** 2))
    err = float(np.mean((ref.astype(np.float64) - approx.astype(np.float64)) ** 2))
    if err <= 1e-18:
        return 100.0
    return round(10.0 * math.log10(sig / err), 3)


def pack_words_128bit_i8(arr: np.ndarray) -> list[str]:
    """Pack 2D int8 array [rows, cols] (cols % 16 == 0) into 128-bit hex strings (32 hex chars).
    Lane 0 (col%16 == 0) goes into bits [7:0] (rightmost 2 hex chars).
    """
    rows, cols = arr.shape
    assert cols % 16 == 0
    lines = []
    u8 = arr.view(np.uint8)
    for r in range(rows):
        for c0 in range(0, cols, 16):
            val = 0
            for lane in range(16):
                val |= int(u8[r, c0 + lane]) << (8 * lane)
            lines.append(f"{val:032x}")
    return lines


def pack_words_128bit_w4(arr_i4: np.ndarray) -> list[str]:
    """Pack 2D signed int4 array [-8..7] [rows, cols] (cols % 32 == 0) into 128-bit hex strings.
    Each 128-bit word stores 32 weights:
      bits [63:0]   = 16 weights (lanes 0..15), 4 bits per lane (lane 0 in [3:0], lane 1 in [7:4], ...)
      bits [127:64] = 16 weights (lanes 16..31), 4 bits per lane.
    """
    rows, cols = arr_i4.shape
    assert cols % 32 == 0
    lines = []
    for r in range(rows):
        for c0 in range(0, cols, 32):
            val = 0
            for lane in range(32):
                nib = int(arr_i4[r, c0 + lane]) & 0xF
                val |= nib << (4 * lane)
            lines.append(f"{val:032x}")
    return lines


def pack_params_64bit(bias: np.ndarray, m16: np.ndarray, shift: np.ndarray) -> list[str]:
    """Pack [bias_i32 (31:0), m16 (47:32), shift (55:48), flags=0 (63:56)] into 64-bit hex strings."""
    lines = []
    for i in range(len(bias)):
        b_u32 = int(np.int32(bias[i])) & 0xFFFFFFFF
        m_u16 = int(m16[i]) & 0xFFFF
        s_u8 = int(shift[i]) & 0xFF
        word = b_u32 | (m_u16 << 32) | (s_u8 << 48)
        lines.append(f"{word:016x}")
    return lines


def write_hex_bytes(path: Path, arr_i8: np.ndarray) -> str:
    u8 = arr_i8.reshape(-1).view(np.uint8)
    path.write_text("\n".join(f"{int(b):02x}" for b in u8) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_hex_lines(path: Path, lines: list[str]) -> str:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_ffn(w_bits: int, folder_name: str, rng_seed_offset: int) -> dict:
    rng = np.random.default_rng(SEED + rng_seed_offset)
    out_dir = GOLDEN_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Realistic ZipFormer FFN activations & weights
    x_f = rng.normal(0.0, 0.70, size=(T_FRAMES, K_DIM)).astype(np.float64)
    w1_f = rng.normal(0.0, 0.14, size=(H_DIM, K_DIM)).astype(np.float64)
    b1_f = rng.normal(0.0, 0.04, size=(H_DIM,)).astype(np.float64)
    w2_f = rng.normal(0.0, 0.11, size=(K_DIM, H_DIM)).astype(np.float64)
    b2_f = rng.normal(0.0, 0.03, size=(K_DIM,)).astype(np.float64)

    # FP32 reference
    h_pre_ref = x_f @ w1_f.T + b1_f
    h_post_ref = swoosh_l_fp32(h_pre_ref)
    y_ref = h_post_ref @ w2_f.T + b2_f

    # Quantization scales (p99.9 / symmetric)
    sx = float(np.max(np.abs(x_f))) / 127.0
    x_q = np.clip(np.rint(x_f / sx), -128, 127).astype(np.int8)

    w_qmax = 127.0 if w_bits == 8 else 7.0
    w_qmin = -128 if w_bits == 8 else -8

    sw1 = np.maximum(np.max(np.abs(w1_f), axis=1) / w_qmax, 1e-8)
    w1_q = np.clip(np.rint(w1_f / sw1[:, None]), w_qmin, int(w_qmax)).astype(np.int8)

    sh_pre = max(float(np.max(np.abs(h_pre_ref))) / 127.0, 1e-5)
    sh_post = max(float(np.max(np.abs(h_post_ref))) / 127.0, 1e-5)
    swoosh_lut = build_lut_256(swoosh_l_fp32, sh_pre, sh_post)

    b1_q = np.rint(b1_f / (sx * sw1)).astype(np.int32)
    m16_1 = np.zeros(H_DIM, dtype=np.int32)
    sh_1 = np.zeros(H_DIM, dtype=np.int32)
    for j in range(H_DIM):
        m16_1[j], sh_1[j] = make_dyadic_m16((sx * sw1[j]) / sh_pre)

    # Layer 1 integer execution
    acc1 = x_q.astype(np.int32) @ w1_q.astype(np.int32).T + b1_q[None, :]
    h_pre_q = dyadic_requantize_round_half_up(acc1, m16_1, sh_1)
    h_post_q = apply_lut_256(h_pre_q, swoosh_lut)

    # Layer 2 quantization
    sw2 = np.maximum(np.max(np.abs(w2_f), axis=1) / w_qmax, 1e-8)
    w2_q = np.clip(np.rint(w2_f / sw2[:, None]), w_qmin, int(w_qmax)).astype(np.int8)
    sy = max(float(np.max(np.abs(y_ref))) / 127.0, 1e-5)
    b2_q = np.rint(b2_f / (sh_post * sw2)).astype(np.int32)
    m16_2 = np.zeros(K_DIM, dtype=np.int32)
    sh_2 = np.zeros(K_DIM, dtype=np.int32)
    for j in range(K_DIM):
        m16_2[j], sh_2[j] = make_dyadic_m16((sh_post * sw2[j]) / sy)

    acc2 = h_post_q.astype(np.int32) @ w2_q.astype(np.int32).T + b2_q[None, :]
    y_q = dyadic_requantize_round_half_up(acc2, m16_2, sh_2)

    sqnr_l1 = sqnr_db(h_post_ref, h_post_q.astype(np.float64) * sh_post)
    sqnr_l2 = sqnr_db(y_ref, y_q.astype(np.float64) * sy)

    # Write hex artifacts
    hashes = {}
    hashes["act_words"] = write_hex_lines(out_dir / "act_words_128b.hex", pack_words_128bit_i8(x_q))
    if w_bits == 8:
        w_lines = pack_words_128bit_i8(w1_q) + pack_words_128bit_i8(w2_q)
    else:
        w_lines = pack_words_128bit_w4(w1_q) + pack_words_128bit_w4(w2_q)
    hashes["wgt_words"] = write_hex_lines(out_dir / "wgt_words_128b.hex", w_lines)
    p_lines = pack_params_64bit(b1_q, m16_1, sh_1) + pack_params_64bit(b2_q, m16_2, sh_2)
    hashes["param_words"] = write_hex_lines(out_dir / "param_words_64b.hex", p_lines)
    hashes["swoosh_lut"] = write_hex_bytes(out_dir / "swoosh_l_lut.hex", swoosh_lut)
    hashes["h_pre_golden"] = write_hex_bytes(out_dir / "h_pre_frames0_15.hex", h_pre_q)
    hashes["h_post_golden"] = write_hex_bytes(out_dir / "h_post_frames0_15.hex", h_post_q)
    hashes["y_golden"] = write_hex_bytes(out_dir / "y_golden_frames0_15.hex", y_q)
    hashes["y_golden_words"] = write_hex_lines(out_dir / "y_golden_words_128b.hex", pack_words_128bit_i8(y_q))
    hashes["h_post_words"] = write_hex_lines(out_dir / "h_post_words_128b.hex", pack_words_128bit_i8(h_post_q))

    np.savez(
        out_dir / "golden_vectors.npz",
        x_q=x_q, w1_q=w1_q, b1_q=b1_q, m16_1=m16_1, sh_1=sh_1,
        w2_q=w2_q, b2_q=b2_q, m16_2=m16_2, sh_2=sh_2,
        swoosh_lut=swoosh_lut, h_pre_q=h_pre_q, h_post_q=h_post_q, y_q=y_q,
    )

    meta = {
        "workload": folder_name,
        "w_bits": w_bits,
        "dims": {"T": T_FRAMES, "K": K_DIM, "H": H_DIM},
        "sqnr_layer1_post_swoosh_db": sqnr_l1,
        "sqnr_layer2_out_db": sqnr_l2,
        "wgt_words_128b_count": len(w_lines),
        "param_words_64b_count": len(p_lines),
        "sha256": hashes,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def generate_joiner() -> dict:
    rng = np.random.default_rng(SEED + 100)
    out_dir = GOLDEN_DIR / "joiner_W8A8"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_f = rng.normal(0.0, 0.65, size=(T_FRAMES, K_DIM)).astype(np.float64)
    dec_f = rng.normal(0.0, 0.45, size=(1, K_DIM)).astype(np.float64)
    wj_f = rng.normal(0.0, 0.16, size=(V_DIM, K_DIM)).astype(np.float64)
    bj_f = rng.normal(0.0, 0.04, size=(V_DIM,)).astype(np.float64)

    sum_ref = enc_f + dec_f
    act_ref = np.tanh(sum_ref)
    logit_ref = act_ref @ wj_f.T + bj_f

    # Shared scale for enc + dec so integer elementwise addition is exact
    s_ed = max(float(np.max(np.abs(enc_f))), float(np.max(np.abs(dec_f))), float(np.max(np.abs(sum_ref)))) / 127.0
    enc_q = np.clip(np.rint(enc_f / s_ed), -128, 127).astype(np.int8)
    dec_q = np.clip(np.rint(dec_f / s_ed), -128, 127).astype(np.int8)

    s_tanh = 1.0 / 127.0
    tanh_lut = build_lut_256(np.tanh, s_ed, s_tanh)

    sum_i16 = enc_q.astype(np.int16) + dec_q.astype(np.int16)
    sum_clamped = np.clip(sum_i16, -128, 127).astype(np.int8)
    j_act_q = apply_lut_256(sum_clamped, tanh_lut)

    swj = np.maximum(np.max(np.abs(wj_f), axis=1) / 127.0, 1e-8)
    wj_q = np.clip(np.rint(wj_f / swj[:, None]), -128, 127).astype(np.int8)
    s_logit = max(float(np.max(np.abs(logit_ref))) / 127.0, 1e-5)
    bj_q = np.rint(bj_f / (s_tanh * swj)).astype(np.int32)
    m16_j = np.zeros(V_DIM, dtype=np.int32)
    sh_j = np.zeros(V_DIM, dtype=np.int32)
    for j in range(V_DIM):
        m16_j[j], sh_j[j] = make_dyadic_m16((s_tanh * swj[j]) / s_logit)

    acc_j = j_act_q.astype(np.int32) @ wj_q.astype(np.int32).T + bj_q[None, :]
    logit_q = dyadic_requantize_round_half_up(acc_j, m16_j, sh_j)

    sqnr_j = sqnr_db(logit_ref, logit_q.astype(np.float64) * s_logit)
    top1_match = float(np.mean(np.argmax(logit_ref, axis=1) == np.argmax(logit_q, axis=1)) * 100.0)

    hashes = {}
    hashes["enc_words"] = write_hex_lines(out_dir / "enc_words_128b.hex", pack_words_128bit_i8(enc_q))
    hashes["dec_words"] = write_hex_lines(out_dir / "dec_words_128b.hex", pack_words_128bit_i8(dec_q))
    hashes["wgt_words"] = write_hex_lines(out_dir / "wgt_words_128b.hex", pack_words_128bit_i8(wj_q))
    hashes["param_words"] = write_hex_lines(out_dir / "param_words_64b.hex", pack_params_64bit(bj_q, m16_j, sh_j))
    hashes["tanh_lut"] = write_hex_bytes(out_dir / "tanh_lut.hex", tanh_lut)
    hashes["j_act_golden"] = write_hex_bytes(out_dir / "j_act_frames0_15.hex", j_act_q)
    hashes["logit_golden"] = write_hex_bytes(out_dir / "logit_golden_frames0_15.hex", logit_q)
    hashes["logit_golden_words"] = write_hex_lines(out_dir / "logit_golden_words_128b.hex", pack_words_128bit_i8(logit_q))

    np.savez(
        out_dir / "golden_vectors.npz",
        enc_q=enc_q, dec_q=dec_q, wj_q=wj_q, bj_q=bj_q, m16_j=m16_j, sh_j=sh_j,
        tanh_lut=tanh_lut, j_act_q=j_act_q, logit_q=logit_q,
    )

    meta = {
        "workload": "joiner_W8A8",
        "w_bits": 8,
        "dims": {"T": T_FRAMES, "D": K_DIM, "V": V_DIM},
        "sqnr_logit_db": sqnr_j,
        "top1_token_match_pct": round(top1_match, 2),
        "wgt_words_128b_count": len(pack_words_128bit_i8(wj_q)),
        "param_words_64b_count": len(bj_q),
        "sha256": hashes,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    m_ffn_w8 = generate_ffn(8, "stage0_layer0_ffn1_W8A8", 0)
    m_ffn_w4 = generate_ffn(4, "stage0_layer0_ffn1_W4A8", 1)
    m_joiner = generate_joiner()
    summary = {
        "stage0_layer0_ffn1_W8A8": m_ffn_w8,
        "stage0_layer0_ffn1_W4A8": m_ffn_w4,
        "joiner_W8A8": m_joiner,
    }
    (GOLDEN_DIR / "golden_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
