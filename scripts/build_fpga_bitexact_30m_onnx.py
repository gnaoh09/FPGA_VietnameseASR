#!/usr/bin/env python3
"""Build 100% RTL-Bit-Accurate FPGA Integer-Datapath Zipformer-30M ONNX Encoders.

Calibrates static p99.99 activation scales across all 36 FFN blocks using
50 held-out VIVOS train utterances (zero leakage into VIVOS test / FLEURS),
then rewrites every FFN block into the exact integer hardware pipeline of
`rtl/zipformer_engine.v`:
  - Input static INT8 quantization: clamp(round(x / sx), -128, 127)
  - Layer 1 integer GEMM (W8 [-127..127] or W4 [-7..7]) + INT32 bias
  - Dyadic M16 * 2^-S round-half-up (exact FLOAT64 Floor -> 0 LSB diff vs Verilog >>>)
  - Clamp [-128, 127] -> 256-entry INT8 SwooshL LUT via Gather
  - Layer 2 integer GEMM + INT32 bias + Dyadic M16 * 2^-S + Clamp [-128, 127] -> * sy_out
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from onnx import TensorProto, helper, numpy_helper

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))
from gen_golden_vectors import build_lut_256, make_dyadic_m16

MODEL_DIR = ROOT_DIR / "data" / "zipformer_30m"
CALIB_DIR = ROOT_DIR / "data" / "vivos" / "train" / "waves"
QUANT_DIR = ROOT_DIR / "build" / "quant_models"


def swoosh_l_zipformer2(x: np.ndarray) -> np.ndarray:
    """Exact SwooshL used in Zipformer-30M FFN ONNX: log(1 + exp(x - 4.0)) - 0.08*x - 0.035."""
    z = x - 4.0
    softplus = np.where(z > 20.0, z, np.log1p(np.exp(np.clip(z, -40.0, 20.0))))
    return softplus - 0.08 * x - 0.035


def compute_fbank(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = float(sr)
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400.0
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(sr, wav.tolist())
    fb.input_finished()
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)], axis=0).astype(np.float32)


def discover_ffn_blocks(model: onnx.ModelProto) -> list[dict]:
    by_name = {n.name: n for n in model.graph.node}
    init_names = {i.name for i in model.graph.initializer}
    blocks = []
    for mm1 in model.graph.node:
        if mm1.op_type == "MatMul" and "feed_forward" in mm1.name and "/in_proj/" in mm1.name:
            prefix = mm1.name.rsplit("/in_proj/MatMul", 1)[0]
            add1 = by_name[prefix + "/in_proj/Add"]
            sub2 = by_name[prefix + "/out_proj/Sub_2"]
            mm2 = by_name[prefix + "/out_proj/MatMul"]
            add2 = by_name[prefix + "/out_proj/Add_1"]
            b1_name = add1.input[0] if add1.input[0] in init_names else add1.input[1]
            b2_name = add2.input[0] if add2.input[0] in init_names else add2.input[1]
            blocks.append({
                "prefix": prefix,
                "x_in": mm1.input[0],
                "w1_name": mm1.input[1],
                "b1_name": b1_name,
                "h_pre": add1.output[0],
                "h_post": sub2.output[0],
                "w2_name": mm2.input[1],
                "b2_name": b2_name,
                "y_out": add2.output[0],
            })
    assert len(blocks) == 36, f"Expected 36 FFN blocks, got {len(blocks)}"
    return blocks


def calibrate_scales(src_onnx: Path, blocks: list[dict], max_utts: int = 50) -> dict[str, dict]:
    """Calibrate p99.99 activation scales on held-out VIVOS train audio."""
    calib_cache = QUANT_DIR / "ffn_calib_scales_vivos_train50.json"
    if calib_cache.exists():
        return json.loads(calib_cache.read_text(encoding="utf-8"))

    probe_model = onnx.load(str(src_onnx))
    orig_outs = {o.name for o in probe_model.graph.output}
    tap_names = []
    for blk in blocks:
        for k in ["x_in", "h_pre", "h_post", "y_out"]:
            tname = blk[k]
            if tname not in orig_outs and tname not in tap_names:
                tap_names.append(tname)
                probe_model.graph.output.append(helper.make_tensor_value_info(tname, TensorProto.FLOAT, None))

    probe_tmp = QUANT_DIR / "_calib_probe.tmp.onnx"
    onnx.save(probe_model, str(probe_tmp))
    sess = ort.InferenceSession(str(probe_tmp), providers=["CPUExecutionProvider"])
    probe_tmp.unlink(missing_ok=True)

    out_names = [o.name for o in sess.get_outputs()]
    name_to_idx = {n: i for i, n in enumerate(out_names)}

    wavs = sorted(CALIB_DIR.rglob("*.wav"))[:max_utts]
    assert len(wavs) >= 20, f"Too few calibration wavs: {len(wavs)}"

    samples: dict[str, list[np.ndarray]] = {t: [] for t in tap_names}
    for wp in wavs:
        audio, sr = sf.read(wp, dtype="float32")
        feats = compute_fbank(audio, sr)
        outs = sess.run(None, {"x": feats[None], "x_lens": np.array([len(feats)], np.int64)})
        for t in tap_names:
            arr = np.abs(outs[name_to_idx[t]]).reshape(-1)
            samples[t].append(arr)

    scales = {}
    for blk in blocks:
        def q9999(tname: str, div: float) -> float:
            vals = np.concatenate(samples[tname])
            return max(float(np.quantile(vals, 0.9999)) / div, 1e-6)

        scales[blk["prefix"]] = {
            "sx": q9999(blk["x_in"], 127.0),
            "s_hpre": q9999(blk["h_pre"], 120.0),
            "s_hpost": q9999(blk["h_post"], 120.0),
            "sy_out": q9999(blk["y_out"], 120.0),
        }

    calib_cache.write_text(json.dumps(scales, indent=2), encoding="utf-8")
    return scales


def build_bitexact_model(src_onnx: Path, dst_onnx: Path, w_bits: int, scales: dict[str, dict]) -> dict:
    """Rewrite all 36 FFN blocks into RTL-exact FPGA integer ONNX subgraphs."""
    meta_path = dst_onnx.with_suffix(".meta.json")
    if dst_onnx.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    model = onnx.load(str(src_onnx))
    blocks = discover_ffn_blocks(model)
    init_by_name = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    blk_by_in_mm = {b["prefix"] + "/in_proj/MatMul": b for b in blocks}

    # Build map of internal FFN nodes to strip and replace cleanly at add2 position
    ffn_prefixes = {b["prefix"] for b in blocks}
    new_nodes = []
    new_inits = list(model.graph.initializer)

    # Shared scalar constants in FLOAT64
    shared_prefix = "_fpga_shared_"
    shared_scalars = {
        f"{shared_prefix}cmin": np.array(-128.0, dtype=np.float64),
        f"{shared_prefix}cmax": np.array(127.0, dtype=np.float64),
        f"{shared_prefix}c128": np.array(128.0, dtype=np.float64),
        f"{shared_prefix}cmin32": np.array(-128.0, dtype=np.float32),
        f"{shared_prefix}cmax32": np.array(127.0, dtype=np.float32),
    }
    for sname, sarr in shared_scalars.items():
        new_inits.append(numpy_helper.from_array(sarr, name=sname))

    qmax = 127.0 if w_bits == 8 else 7.0
    max_abs_bias_q = 0
    total_ffn_params = 0
    w_sqnr_sig, w_sqnr_err = 0.0, 0.0

    for node in model.graph.node:
        matched_prefix = None
        for p in ffn_prefixes:
            if node.name.startswith(p + "/"):
                matched_prefix = p
                break

        if matched_prefix is None:
            new_nodes.append(node)
            continue

        # Only emit replacement subgraph when encountering `.../out_proj/Add_1` (the final FFN output node)
        if node.name != matched_prefix + "/out_proj/Add_1":
            continue

        blk = blk_by_in_mm[matched_prefix + "/in_proj/MatMul"]
        sc = scales[matched_prefix]
        sx, s_hpre, s_hpost, sy_out = sc["sx"], sc["s_hpre"], sc["s_hpost"], sc["sy_out"]

        w1_f = init_by_name[blk["w1_name"]].astype(np.float64)
        b1_f = init_by_name[blk["b1_name"]].astype(np.float64)
        w2_f = init_by_name[blk["w2_name"]].astype(np.float64)
        b2_f = init_by_name[blk["b2_name"]].astype(np.float64)
        total_ffn_params += w1_f.size + w2_f.size

        # Quantize W1 [in, mid] per output col (axis=1)
        sw1 = np.maximum(np.max(np.abs(w1_f), axis=0), 1e-8) / qmax
        w1_q = np.clip(np.rint(w1_f / sw1[None, :]), -qmax, qmax)
        b1_q = np.rint(b1_f / (sx * sw1))
        max_abs_bias_q = max(max_abs_bias_q, int(np.max(np.abs(b1_q))))

        m1_arr = np.empty(w1_f.shape[1], dtype=np.int64)
        sh1_arr = np.empty(w1_f.shape[1], dtype=np.int64)
        for c in range(w1_f.shape[1]):
            m1_arr[c], sh1_arr[c] = make_dyadic_m16((sx * float(sw1[c])) / s_hpre)

        lut_i8 = build_lut_256(swoosh_l_zipformer2, s_hpre, s_hpost)
        lut_f64 = np.array([float(lut_i8[(i - 128) & 0xFF]) for i in range(256)], dtype=np.float64)

        sw2 = np.maximum(np.max(np.abs(w2_f), axis=0), 1e-8) / qmax
        w2_q = np.clip(np.rint(w2_f / sw2[None, :]), -qmax, qmax)
        b2_q = np.rint(b2_f / (s_hpost * sw2))
        max_abs_bias_q = max(max_abs_bias_q, int(np.max(np.abs(b2_q))))

        w_sqnr_sig += float(np.sum(w1_f**2) + np.sum(w2_f**2))
        w_sqnr_err += float(np.sum((w1_f - w1_q * sw1[None, :])**2) + np.sum((w2_f - w2_q * sw2[None, :])**2))

        m2_arr = np.empty(w2_f.shape[1], dtype=np.int64)
        sh2_arr = np.empty(w2_f.shape[1], dtype=np.int64)
        for c in range(w2_f.shape[1]):
            m2_arr[c], sh2_arr[c] = make_dyadic_m16((s_hpost * float(sw2[c])) / sy_out)

        tag = matched_prefix.strip("/").replace("/", "_").replace(".", "_")
        pfx = f"_fpga_{tag}_"

        local_inits = {
            "inv_sx": np.array(1.0 / sx, dtype=np.float32),
            "sy_out": np.array(sy_out, dtype=np.float64),
            "w1_q": w1_q.astype(np.float64),
            "b1_q": b1_q.astype(np.float64),
            "m1": m1_arr.astype(np.float64),
            "half1": np.where(sh1_arr > 0, 2.0 ** (sh1_arr - 1), 0.0),
            "pow2_1": 2.0 ** sh1_arr,
            "lut": lut_f64,
            "w2_q": w2_q.astype(np.float64),
            "b2_q": b2_q.astype(np.float64),
            "m2": m2_arr.astype(np.float64),
            "half2": np.where(sh2_arr > 0, 2.0 ** (sh2_arr - 1), 0.0),
            "pow2_2": 2.0 ** sh2_arr,
        }
        for k, arr in local_inits.items():
            new_inits.append(numpy_helper.from_array(arr, name=pfx + k))

        # Wire RTL integer pipeline in FLOAT64 for exact 53-bit integer arithmetic + negative Floor match to >>>
        cmin, cmax, c128 = f"{shared_prefix}cmin", f"{shared_prefix}cmax", f"{shared_prefix}c128"
        cmin32, cmax32 = f"{shared_prefix}cmin32", f"{shared_prefix}cmax32"
        new_nodes.extend([
            helper.make_node("Mul", [blk["x_in"], pfx + "inv_sx"], [pfx + "xs"]),
            helper.make_node("Round", [pfx + "xs"], [pfx + "xr"]),
            helper.make_node("Clip", [pfx + "xr", cmin32, cmax32], [pfx + "xq32"]),
            helper.make_node("Cast", [pfx + "xq32"], [pfx + "xq"], to=TensorProto.DOUBLE),
            # Layer 1 GEMM + dyadic round-half-up
            helper.make_node("MatMul", [pfx + "xq", pfx + "w1_q"], [pfx + "mm1"]),
            helper.make_node("Add", [pfx + "mm1", pfx + "b1_q"], [pfx + "acc1"]),
            helper.make_node("Mul", [pfx + "acc1", pfx + "m1"], [pfx + "prod1"]),
            helper.make_node("Add", [pfx + "prod1", pfx + "half1"], [pfx + "rnd1"]),
            helper.make_node("Div", [pfx + "rnd1", pfx + "pow2_1"], [pfx + "div1"]),
            helper.make_node("Floor", [pfx + "div1"], [pfx + "fl1"]),
            helper.make_node("Clip", [pfx + "fl1", cmin, cmax], [pfx + "hpre_q"]),
            # 256-entry INT8 SwooshL LUT
            helper.make_node("Add", [pfx + "hpre_q", c128], [pfx + "idx_f"]),
            helper.make_node("Cast", [pfx + "idx_f"], [pfx + "idx"], to=TensorProto.INT64),
            helper.make_node("Gather", [pfx + "lut", pfx + "idx"], [pfx + "hpost_q"], axis=0),
            # Layer 2 GEMM + dyadic round-half-up -> y_out (FLOAT32)
            helper.make_node("MatMul", [pfx + "hpost_q", pfx + "w2_q"], [pfx + "mm2"]),
            helper.make_node("Add", [pfx + "mm2", pfx + "b2_q"], [pfx + "acc2"]),
            helper.make_node("Mul", [pfx + "acc2", pfx + "m2"], [pfx + "prod2"]),
            helper.make_node("Add", [pfx + "prod2", pfx + "half2"], [pfx + "rnd2"]),
            helper.make_node("Div", [pfx + "rnd2", pfx + "pow2_2"], [pfx + "div2"]),
            helper.make_node("Floor", [pfx + "div2"], [pfx + "fl2"]),
            helper.make_node("Clip", [pfx + "fl2", cmin, cmax], [pfx + "yq"]),
            helper.make_node("Mul", [pfx + "yq", pfx + "sy_out"], [pfx + "yf64"]),
            helper.make_node("Cast", [pfx + "yf64"], [blk["y_out"]], to=TensorProto.FLOAT),
        ])

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    used_inputs = {inp for n in new_nodes for inp in n.input}
    pruned_inits = [i for i in new_inits if i.name in used_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(pruned_inits)
    del model.graph.value_info[:]  # Clear stale shape/type cache from old nodes
    model.ir_version = min(model.ir_version, 9)

    onnx.save(model, str(dst_onnx))
    # Quick runtime smoke check
    sess = ort.InferenceSession(str(dst_onnx), providers=["CPUExecutionProvider"])
    dummy_x = np.zeros((1, 40, 80), dtype=np.float32)
    dummy_l = np.array([40], dtype=np.int64)
    sess.run(None, {"x": dummy_x, "x_lens": dummy_l})

    import math
    w_sqnr_db = round(10.0 * math.log10(w_sqnr_sig / max(w_sqnr_err, 1e-18)), 2)
    meta = {
        "w_bits": w_bits,
        "ffn_blocks_rewritten": len(blocks),
        "params_quantized_M": round(total_ffn_params / 1e6, 3),
        "weight_sqnr_db": w_sqnr_db,
        "max_abs_int32_bias": max_abs_bias_q,
        "onnx_size_MB": round(dst_onnx.stat().st_size / (1024 * 1024), 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    src = MODEL_DIR / "encoder-epoch-20-avg-10.onnx"
    model = onnx.load(str(src), load_external_data=False)
    blocks = discover_ffn_blocks(model)
    scales = calibrate_scales(src, blocks, max_utts=50)
    print(f"Calibrated {len(scales)} FFN blocks cleanly on VIVOS train.")
    for wb in [8, 4]:
        dst = QUANT_DIR / f"encoder_fpga_bitexact_w{wb}a8.onnx"
        meta = build_bitexact_model(src, dst, w_bits=wb, scales=scales)
        print(f"Built {dst.name}: {meta}")


if __name__ == "__main__":
    main()
