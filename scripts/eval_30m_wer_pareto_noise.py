#!/usr/bin/env python3
"""Evaluate REAL Vietnamese ASR WER / CER / Tone-Loss & Paired Bootstrap 95% CI
using hynt/Zipformer-30M-RNNT-6000h across VIVOS Test (760 utts) and Google FLEURS vi_vn (857 utts):
  1. Baseline & Local ARM64 CPU (Apple M5 1T/4T): FP32 & W8A8 (int8.onnx)
  2. True FPGA Bit-Exact Integer Datapath (static p99.99 A8 + dyadic M16*2^-S + 256-LUT SwooshL):
       - 30M_FPGA_BitExact_FFN_W8A8
       - 30M_FPGA_BitExact_FFN_W4A8
  3. Weight Quantization Pareto: FFN W4/W3/W2 vs ALL-MatMul W4/W3/W2
  4. Additive White Gaussian Noise robustness on VIVOS (Clean, SNR 20/10/5 dB)
  5. Exact Paired Bootstrap (B=2000, seed=20261117; cluster-paired on FLEURS) for ΔWER & ΔToneGap.
"""

from __future__ import annotations

import json
import math
import platform
import re
import resource
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jiwer
import numpy as np
import onnx
import sherpa_onnx
import soundfile as sf
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))
from build_fpga_bitexact_30m_onnx import build_bitexact_model, calibrate_scales, discover_ffn_blocks

MODEL_DIR = ROOT_DIR / "data" / "zipformer_30m"
VIVOS_TEST_DIR = ROOT_DIR / "data" / "vivos" / "test"
FLEURS_TSV = ROOT_DIR / "data" / "fleurs_hf" / "data" / "vi_vn" / "test.tsv"
FLEURS_WAV_DIR = ROOT_DIR / "data" / "fleurs_vi_test"
QUANT_DIR = ROOT_DIR / "build" / "quant_models"
BUILD_DIR = ROOT_DIR / "build"

NOISE_SEED = 20261117
BOOT_SEED = 20261117
BOOT_B = 2000
TONE_COMBINING_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}


def norm_vi(text: str) -> str:
    t = re.sub(r"[^\w\s]", " ", text.upper())
    return " ".join(t.split())


def strip_tones_only(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    kept = "".join(ch for ch in nfd if ch not in TONE_COMBINING_MARKS)
    return unicodedata.normalize("NFC", kept)


def strip_all_diacritics(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    kept = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return kept.replace("Đ", "D").replace("đ", "d")


def _choose_group_size(in_dim: int, max_g: int = 64) -> int:
    for g in range(min(max_g, in_dim), 0, -1):
        if in_dim % g == 0:
            return g
    return 1


def quantize_weight_symmetric_per_channel(w: np.ndarray, bits: int) -> np.ndarray:
    qmax = float((1 << (bits - 1)) - 1)
    w_f = w.astype(np.float32)
    scale = np.maximum(np.max(np.abs(w_f), axis=0, keepdims=True), 1e-8) / qmax
    w_q = np.clip(np.rint(w_f / scale), -qmax, qmax)
    return (w_q * scale).astype(w.dtype)


def quantize_weight_groupwise_symmetric(w: np.ndarray, bits: int, group_size: int = 64) -> np.ndarray:
    in_dim, out_dim = w.shape
    g = _choose_group_size(in_dim, group_size)
    qmax = float((1 << (bits - 1)) - 1)
    w_f = w.astype(np.float32).reshape(in_dim // g, g, out_dim)
    scale = np.maximum(np.max(np.abs(w_f), axis=1, keepdims=True), 1e-8) / qmax
    w_q = np.clip(np.rint(w_f / scale), -qmax, qmax)
    return (w_q * scale).reshape(in_dim, out_dim).astype(w.dtype)


def quantize_weight_groupwise_uniform_minmax(w: np.ndarray, bits: int, group_size: int = 64) -> np.ndarray:
    in_dim, out_dim = w.shape
    g = _choose_group_size(in_dim, group_size)
    n_levels = float((1 << bits) - 1)
    w_f = w.astype(np.float32).reshape(in_dim // g, g, out_dim)
    w_min = np.min(w_f, axis=1, keepdims=True)
    w_max = np.max(w_f, axis=1, keepdims=True)
    scale = np.maximum((w_max - w_min) / n_levels, 1e-8)
    w_q = np.clip(np.rint((w_f - w_min) / scale), 0.0, n_levels)
    return (w_q * scale + w_min).reshape(in_dim, out_dim).astype(w.dtype)


def build_quantized_encoder(src_onnx: Path, dst_onnx: Path, scope: str, quant_fn, dynamic_a8: bool = False) -> dict:
    meta_p = dst_onnx.with_suffix(".meta.json")
    if dst_onnx.exists() and meta_p.exists():
        return json.loads(meta_p.read_text(encoding="utf-8"))

    model = onnx.load(str(src_onnx))
    init_by_name = {i.name: i for i in model.graph.initializer}
    target_weights = set()
    total_matmul_params = 0
    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) >= 2 and node.input[1] in init_by_name:
            w_arr = numpy_helper.to_array(init_by_name[node.input[1]])
            if w_arr.ndim == 2:
                total_matmul_params += w_arr.size
                if scope == "all" or (scope == "ffn" and "feed_forward" in node.name):
                    target_weights.add(node.input[1])

    q_params, sig_e, err_e = 0, 0.0, 0.0
    for w_name in target_weights:
        init = init_by_name[w_name]
        w_orig = numpy_helper.to_array(init)
        w_q = quant_fn(w_orig)
        q_params += w_orig.size
        sig_e += float(np.sum(w_orig.astype(np.float64) ** 2))
        err_e += float(np.sum((w_orig.astype(np.float64) - w_q.astype(np.float64)) ** 2))
        init.CopyFrom(numpy_helper.from_array(w_q, name=w_name))

    tmp_path = dst_onnx.with_suffix(".tmp.onnx") if dynamic_a8 else dst_onnx
    onnx.save(model, str(tmp_path))
    if dynamic_a8:
        quantize_dynamic(
            model_input=str(tmp_path),
            model_output=str(dst_onnx),
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["MatMul"],
        )
        tmp_path.unlink(missing_ok=True)

    sqnr_w_db = round(10.0 * math.log10(sig_e / max(err_e, 1e-18)), 2)
    meta = {
        "tensors_quantized": len(target_weights),
        "params_quantized_M": round(q_params / 1e6, 3),
        "matmul_params_total_M": round(total_matmul_params / 1e6, 3),
        "weight_sqnr_db": sqnr_w_db,
        "onnx_size_MB": round(dst_onnx.stat().st_size / (1024 * 1024), 2),
    }
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_vivos_test() -> list[tuple[str, str, np.ndarray, int]]:
    items = []
    for line in (VIVOS_TEST_DIR / "prompts.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id, text = line.split(" ", 1)
        spk = utt_id.split("_")[0]
        audio, sr = sf.read(VIVOS_TEST_DIR / "waves" / spk / f"{utt_id}.wav", dtype="float32")
        items.append((utt_id, norm_vi(text), audio, sr))
    return items


def load_fleurs_test() -> list[tuple[str, str, np.ndarray, int]]:
    wav_map = {p.name: p for p in FLEURS_WAV_DIR.rglob("*.wav")}
    items = []
    for line in FLEURS_TSV.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        cluster_id, wav_name, _raw, norm_txt = cols[0], cols[1], cols[2], cols[3]
        audio, sr = sf.read(wav_map[wav_name], dtype="float32")
        items.append((f"fleurs_{cluster_id}", norm_vi(norm_txt), audio, sr))
    return items


def add_awgn(audio: np.ndarray, snr_db: float, idx: int) -> np.ndarray:
    sig_power = float(np.mean(audio.astype(np.float64) ** 2))
    if sig_power <= 1e-12:
        return audio
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(NOISE_SEED + idx)
    noise = rng.normal(0.0, math.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def eval_recognizer(
    dataset_name: str,
    model_name: str,
    encoder_path: Path,
    decoder_path: Path,
    joiner_path: Path,
    dataset: list[tuple[str, str, np.ndarray, int]],
    snr_db: float | None = None,
    num_threads: int = 1,
) -> tuple[dict, dict]:
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=str(MODEL_DIR / "config.json"),
        encoder=str(encoder_path),
        decoder=str(decoder_path),
        joiner=str(joiner_path),
        num_threads=num_threads,
    )
    n = len(dataset)
    wer_err = np.zeros(n, dtype=np.int32)
    wer_ref = np.zeros(n, dtype=np.int32)
    notone_err = np.zeros(n, dtype=np.int32)
    notone_ref = np.zeros(n, dtype=np.int32)
    cer_err = np.zeros(n, dtype=np.int32)
    cer_ref = np.zeros(n, dtype=np.int32)
    clusters = []

    total_audio_s = 0.0
    t0 = time.perf_counter()

    for idx, (cid, ref_norm, audio, sr) in enumerate(dataset):
        wav = audio if snr_db is None else add_awgn(audio, snr_db, idx)
        total_audio_s += len(wav) / sr
        stream = rec.create_stream()
        stream.accept_waveform(sr, wav)
        rec.decode_stream(stream)
        hyp = norm_vi(stream.result.text)

        wo = jiwer.process_words(ref_norm, hyp)
        wer_err[idx] = wo.substitutions + wo.deletions + wo.insertions
        wer_ref[idx] = wo.hits + wo.substitutions + wo.deletions

        nto = jiwer.process_words(strip_tones_only(ref_norm), strip_tones_only(hyp))
        notone_err[idx] = nto.substitutions + nto.deletions + nto.insertions
        notone_ref[idx] = nto.hits + nto.substitutions + nto.deletions

        co = jiwer.process_characters(ref_norm, hyp)
        cer_err[idx] = co.substitutions + co.deletions + co.insertions
        cer_ref[idx] = co.hits + co.substitutions + co.deletions
        clusters.append(cid)

    decode_s = time.perf_counter() - t0
    wer_pct = round(100.0 * float(wer_err.sum()) / float(wer_ref.sum()), 3)
    notone_pct = round(100.0 * float(notone_err.sum()) / float(notone_ref.sum()), 3)
    cer_pct = round(100.0 * float(cer_err.sum()) / float(cer_ref.sum()), 3)

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "snr_db": "clean" if snr_db is None else snr_db,
        "num_threads": num_threads,
        "utterances": n,
        "total_audio_s": round(total_audio_s, 2),
        "decode_wall_s": round(decode_s, 2),
        "rtf": round(decode_s / total_audio_s, 5),
        "wer_pct": wer_pct,
        "cer_pct": cer_pct,
        "wer_no_tone_pct": notone_pct,
        "tone_only_gap_pct": round(wer_pct - notone_pct, 3),
    }
    stats = {
        "wer_err": wer_err,
        "wer_ref": wer_ref,
        "notone_err": notone_err,
        "notone_ref": notone_ref,
        "clusters": clusters,
    }
    return summary, stats


def _worker_job(job: tuple) -> tuple[str, dict, dict]:
    kind, ds_name, name, enc, dec, jn, snr_db, extra = job
    ds = load_vivos_test() if ds_name == "vivos" else load_fleurs_test()
    summary, stats = eval_recognizer(ds_name, name, Path(enc), Path(dec), Path(jn), ds, snr_db=snr_db, num_threads=1)
    summary.update(extra)
    return kind, summary, stats


def paired_bootstrap_delta(
    base_stats: dict,
    cand_stats: dict,
    cluster_grouped: bool = False,
    n_boot: int = BOOT_B,
) -> dict:
    """Exact paired (or cluster-paired) bootstrap for ΔWER and ΔTone-Only/SB [95% CI]."""
    rng = np.random.default_rng(BOOT_SEED)
    b_err, b_ref = base_stats["wer_err"], base_stats["wer_ref"]
    c_err, c_ref = cand_stats["wer_err"], cand_stats["wer_ref"]
    b_ne, b_nr = base_stats["notone_err"], base_stats["notone_ref"]
    c_ne, c_nr = cand_stats["notone_err"], cand_stats["notone_ref"]

    if cluster_grouped:
        clusters = base_stats["clusters"]
        uniq = {}
        for i, cid in enumerate(clusters):
            uniq.setdefault(cid, []).append(i)
        grp_lists = list(uniq.values())
        g = len(grp_lists)
        gb_err = np.array([b_err[ix].sum() for ix in grp_lists], dtype=np.float64)
        gb_ref = np.array([b_ref[ix].sum() for ix in grp_lists], dtype=np.float64)
        gc_err = np.array([c_err[ix].sum() for ix in grp_lists], dtype=np.float64)
        gc_ref = np.array([c_ref[ix].sum() for ix in grp_lists], dtype=np.float64)
        gb_ne = np.array([b_ne[ix].sum() for ix in grp_lists], dtype=np.float64)
        gb_nr = np.array([b_nr[ix].sum() for ix in grp_lists], dtype=np.float64)
        gc_ne = np.array([c_ne[ix].sum() for ix in grp_lists], dtype=np.float64)
        gc_nr = np.array([c_nr[ix].sum() for ix in grp_lists], dtype=np.float64)
    else:
        g = len(b_err)
        gb_err, gb_ref = b_err.astype(np.float64), b_ref.astype(np.float64)
        gc_err, gc_ref = c_err.astype(np.float64), c_ref.astype(np.float64)
        gb_ne, gb_nr = b_ne.astype(np.float64), b_nr.astype(np.float64)
        gc_ne, gc_nr = c_ne.astype(np.float64), c_nr.astype(np.float64)

    d_wer = np.empty(n_boot, dtype=np.float64)
    d_tone = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, g, size=g)
        bw = 100.0 * gb_err[draw].sum() / gb_ref[draw].sum()
        cw = 100.0 * gc_err[draw].sum() / gc_ref[draw].sum()
        bt = bw - 100.0 * gb_ne[draw].sum() / gb_nr[draw].sum()
        ct = cw - 100.0 * gc_ne[draw].sum() / gc_nr[draw].sum()
        d_wer[b] = cw - bw
        d_tone[b] = ct - bt

    lo_w, hi_w = float(np.percentile(d_wer, 2.5)), float(np.percentile(d_wer, 97.5))
    lo_t, hi_t = float(np.percentile(d_tone, 2.5)), float(np.percentile(d_tone, 97.5))
    obs_bw = 100.0 * float(b_err.sum()) / float(b_ref.sum())
    obs_cw = 100.0 * float(c_err.sum()) / float(c_ref.sum())
    obs_bt = obs_bw - 100.0 * float(b_ne.sum()) / float(b_nr.sum())
    obs_ct = obs_cw - 100.0 * float(c_ne.sum()) / float(c_nr.sum())
    return {
        "n_units": g,
        "cluster_grouped": cluster_grouped,
        "delta_wer_pct": round(obs_cw - obs_bw, 3),
        "delta_wer_ci95": [round(lo_w, 3), round(hi_w, 3)],
        "passes_ni_margin_0_10": bool(hi_w <= 0.10),
        "delta_tone_gap_pct": round(obs_ct - obs_bt, 3),
        "delta_tone_ci95": [round(lo_t, 3), round(hi_t, 3)],
        "passes_tone_margin_0_10": bool(hi_t <= 0.10),
    }


def merge_stats(s1: dict, s2: dict) -> dict:
    return {
        "wer_err": np.concatenate([s1["wer_err"], s2["wer_err"]]),
        "wer_ref": np.concatenate([s1["wer_ref"], s2["wer_ref"]]),
        "notone_err": np.concatenate([s1["notone_err"], s2["notone_ref"] * 0 + s2["notone_err"]]),
        "notone_ref": np.concatenate([s1["notone_ref"], s2["notone_ref"]]),
        "clusters": s1["clusters"] + s2["clusters"],
    }


def main() -> None:
    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    enc_fp32 = MODEL_DIR / "encoder-epoch-20-avg-10.onnx"
    dec_fp32 = MODEL_DIR / "decoder-epoch-20-avg-10.onnx"
    jn_fp32 = MODEL_DIR / "joiner-epoch-20-avg-10.onnx"
    enc_int8 = MODEL_DIR / "encoder-epoch-20-avg-10.int8.onnx"
    dec_int8 = MODEL_DIR / "decoder-epoch-20-avg-10.int8.onnx"
    jn_int8 = MODEL_DIR / "joiner-epoch-20-avg-10.int8.onnx"

    # Build/load FPGA Bit-Exact Integer Encoders + weight-quantized encoders
    model_raw = onnx.load(str(enc_fp32), load_external_data=False)
    blocks = discover_ffn_blocks(model_raw)
    scales = calibrate_scales(enc_fp32, blocks, max_utts=50)

    variants = {}
    for wb, key in [(8, "fpga_bitexact_w8a8"), (4, "fpga_bitexact_w4a8")]:
        dst = QUANT_DIR / f"encoder_{key}.onnx"
        meta = build_bitexact_model(enc_fp32, dst, w_bits=wb, scales=scales)
        variants[key] = {"path": dst, **meta}

    configs = [
        ("ffn_w4_perch", "ffn", lambda w: quantize_weight_symmetric_per_channel(w, 4), False),
        ("all_w4_perch", "all", lambda w: quantize_weight_symmetric_per_channel(w, 4), False),
        ("ffn_w4_a8dyn", "ffn", lambda w: quantize_weight_symmetric_per_channel(w, 4), True),
        ("ffn_w3_g64", "ffn", lambda w: quantize_weight_groupwise_symmetric(w, 3, 64), False),
        ("all_w3_g64", "all", lambda w: quantize_weight_groupwise_symmetric(w, 3, 64), False),
        ("ffn_w2_u4_g64", "ffn", lambda w: quantize_weight_groupwise_uniform_minmax(w, 2, 64), False),
        ("ffn_w2_tern_g64", "ffn", lambda w: quantize_weight_groupwise_symmetric(w, 2, 64), False),
        ("all_w2_u4_g64", "all", lambda w: quantize_weight_groupwise_uniform_minmax(w, 2, 64), False),
    ]
    for key, scope, fn, dyn_a8 in configs:
        dst = QUANT_DIR / f"encoder_{key}.onnx"
        meta = build_quantized_encoder(enc_fp32, dst, scope=scope, quant_fn=fn, dynamic_a8=dyn_a8)
        variants[key] = {"path": dst, **meta}

    vivos_ds = load_vivos_test()
    fleurs_ds = load_fleurs_test()
    print(f"Loaded VIVOS={len(vivos_ds)} utts, FLEURS={len(fleurs_ds)} utts", flush=True)

    # Uncontended ARM64 M5 1T/4T baseline
    vivos_clean_summaries = []
    vivos_stats_map = {}
    for label, enc, dec, jn, nthr in [
        ("30M_FP32", enc_fp32, dec_fp32, jn_fp32, 1),
        ("30M_FP32_4T", enc_fp32, dec_fp32, jn_fp32, 4),
        ("30M_W8A8", enc_int8, dec_int8, jn_int8, 1),
        ("30M_W8A8_4T", enc_int8, dec_int8, jn_int8, 4),
    ]:
        s, st = eval_recognizer("vivos", label, enc, dec, jn, vivos_ds, snr_db=None, num_threads=nthr)
        vivos_clean_summaries.append(s)
        if nthr == 1:
            vivos_stats_map[label] = st
        print("[VIVOS-BASE]", s, flush=True)

    # Parallel evaluation jobs
    jobs = []
    for key, info in variants.items():
        use_i8_dj = "a8dyn" in key
        mname = f"30M_{key}"
        jobs.append((
            "vivos_clean", "vivos", mname, str(info["path"]),
            str(dec_int8 if use_i8_dj else dec_fp32),
            str(jn_int8 if use_i8_dj else jn_fp32),
            None,
            {"weight_sqnr_db": info["weight_sqnr_db"], "params_quantized_M": info["params_quantized_M"]},
        ))

    noise_models = [
        ("30M_FP32", enc_fp32, dec_fp32, jn_fp32),
        ("30M_W8A8", enc_int8, dec_int8, jn_int8),
        ("30M_fpga_bitexact_w4a8", variants["fpga_bitexact_w4a8"]["path"], dec_fp32, jn_fp32),
        ("30M_ffn_w4_perch", variants["ffn_w4_perch"]["path"], dec_fp32, jn_fp32),
        ("30M_all_w4_perch", variants["all_w4_perch"]["path"], dec_fp32, jn_fp32),
        ("30M_ffn_w3_g64", variants["ffn_w3_g64"]["path"], dec_fp32, jn_fp32),
        ("30M_ffn_w2_u4_g64", variants["ffn_w2_u4_g64"]["path"], dec_fp32, jn_fp32),
    ]
    for snr in [20, 10, 5]:
        for name, enc, dec, jn in noise_models:
            jobs.append(("vivos_noise", "vivos", name, str(enc), str(dec), str(jn), snr, {}))

    fleurs_models = [
        ("30M_FP32", enc_fp32, dec_fp32, jn_fp32),
        ("30M_W8A8", enc_int8, dec_int8, jn_int8),
        ("30M_fpga_bitexact_w8a8", variants["fpga_bitexact_w8a8"]["path"], dec_fp32, jn_fp32),
        ("30M_fpga_bitexact_w4a8", variants["fpga_bitexact_w4a8"]["path"], dec_fp32, jn_fp32),
        ("30M_ffn_w4_perch", variants["ffn_w4_perch"]["path"], dec_fp32, jn_fp32),
        ("30M_ffn_w3_g64", variants["ffn_w3_g64"]["path"], dec_fp32, jn_fp32),
        ("30M_ffn_w2_u4_g64", variants["ffn_w2_u4_g64"]["path"], dec_fp32, jn_fp32),
    ]
    for name, enc, dec, jn in fleurs_models:
        jobs.append(("fleurs_clean", "fleurs", name, str(enc), str(dec), str(jn), None, {}))

    vivos_noise_summaries = []
    vivos_noise_stats = {}
    fleurs_clean_summaries = []
    fleurs_stats_map = {}

    with ProcessPoolExecutor(max_workers=4) as pool:
        for kind, s, st in pool.map(_worker_job, jobs):
            if kind == "vivos_clean":
                vivos_clean_summaries.append(s)
                vivos_stats_map[s["model"]] = st
                print("[VIVOS-QUANT]", s, flush=True)
            elif kind == "vivos_noise":
                vivos_noise_summaries.append(s)
                vivos_noise_stats[(s["snr_db"], s["model"])] = st
                print(f"[VIVOS-NOISE {s['snr_db']}dB]", s, flush=True)
            else:
                fleurs_clean_summaries.append(s)
                fleurs_stats_map[s["model"]] = st
                print("[FLEURS]", s, flush=True)

    # Paired bootstrap CIs (B = 2000, seed = 20261117)
    base_v = vivos_stats_map["30M_FP32"]
    for s in vivos_clean_summaries:
        m = s["model"].replace("_4T", "")
        if m in vivos_stats_map and s["model"] != "30M_FP32":
            s["bootstrap_vs_fp32"] = paired_bootstrap_delta(base_v, vivos_stats_map[m], cluster_grouped=False)

    base_f = fleurs_stats_map["30M_FP32"]
    pooled_results = []
    base_pooled = merge_stats(base_v, base_f)
    for s in fleurs_clean_summaries:
        m = s["model"]
        if m != "30M_FP32":
            s["cluster_bootstrap_vs_fp32"] = paired_bootstrap_delta(base_f, fleurs_stats_map[m], cluster_grouped=True)
        # Combined VIVOS (760 utts) + FLEURS (347 clusters) = 1,617 utterances / 1,107 independent clusters
        cand_pooled = merge_stats(vivos_stats_map[m], fleurs_stats_map[m])
        pw = round(100.0 * float(cand_pooled["wer_err"].sum()) / float(cand_pooled["wer_ref"].sum()), 3)
        pt = pw - round(100.0 * float(cand_pooled["notone_err"].sum()) / float(cand_pooled["notone_ref"].sum()), 3)
        pr = {
            "model": m,
            "utterances": len(cand_pooled["wer_err"]),
            "wer_pct": pw,
            "tone_only_gap_pct": round(pt, 3),
        }
        if m != "30M_FP32":
            pr["cluster_bootstrap_vs_fp32"] = paired_bootstrap_delta(base_pooled, cand_pooled, cluster_grouped=True)
        pooled_results.append(pr)

    for s in vivos_noise_summaries:
        snr, m = s["snr_db"], s["model"]
        if m != "30M_FP32":
            s["bootstrap_vs_fp32"] = paired_bootstrap_delta(
                vivos_noise_stats[(snr, "30M_FP32")], vivos_noise_stats[(snr, m)], cluster_grouped=False
            )

    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)
    out = {
        "model_repo": "hynt/Zipformer-30M-RNNT-6000h",
        "calibration": "50 held-out VIVOS train utterances (p99.99 static scales, 36 FFN blocks, zero test leakage)",
        "datasets": {
            "vivos_test": "760 utterances (2685.54 s)",
            "fleurs_vi_vn_test": "857 utterances across 347 sentence clusters (MCV-vi ungated unavailable; FLEURS cluster-bootstrapped)",
        },
        "bootstrap_config": {"B": BOOT_B, "seed": BOOT_SEED, "ni_upper_margin_pct": 0.10},
        "host_arm64_cpu": f"{platform.machine()} ({platform.platform()})",
        "peak_rss_mb": peak_rss_mb,
        "vivos_clean": vivos_clean_summaries,
        "fleurs_clean": fleurs_clean_summaries,
        "pooled_vivos_plus_fleurs": pooled_results,
        "vivos_noise": vivos_noise_summaries,
    }
    out_path = BUILD_DIR / "wer_30m_v2_bootstrap_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved v2 bootstrap + FPGA bitexact results to {out_path} (Peak RSS = {peak_rss_mb} MB)")


if __name__ == "__main__":
    main()
