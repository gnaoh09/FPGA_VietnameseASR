#!/usr/bin/env python3
"""G5b Real-Hardware 36-FFN Resident Zipformer-30M Execution from 8 MiB SDR SDRAM.

Packs all 36 FFN blocks (72 W4A8 linear layers + 64-bit params) of Zipformer-30M
into a single ~6.64 MiB SDRAM image, uploads it ONCE into HY57V641620FTP-H,
then executes all 36 FFN blocks on real VIVOS speech activations extracted from
`build/quant_models/encoder_fpga_bitexact_w4a8.onnx`, verifying 0 LSB mismatch
on all 36/36 blocks without streaming a single weight over JTAG during inference.
"""

from __future__ import annotations

import argparse
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
from build_fpga_bitexact_30m_onnx import compute_fbank, discover_ffn_blocks
from fpga_jtag_runner import (
    OP_COMMIT_ACT,
    UsbBlasterJtag,
    pack_cfg1,
    pack_cfg2,
)
from gen_golden_vectors import (
    apply_lut_256,
    dyadic_requantize_round_half_up,
    pack_words_128bit_i8,
)
from run_ffn_512_1536_real import pack_cfg0_ext, pack_cfg3_ext
from run_g4b_sdram_experiment import measure_pll_clocks
from sdram_helpers import (
    pack_layer_w4_sdram_words,
    run_sdram_stream_layer,
    write_lut_256,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
SRC_FP32_ONNX = ROOT_DIR / "data" / "zipformer_30m" / "encoder-epoch-20-avg-10.onnx"
BITEXACT_W4_ONNX = ROOT_DIR / "build" / "quant_models" / "encoder_fpga_bitexact_w4a8.onnx"
DEFAULT_WAV = ROOT_DIR / "data" / "vivos" / "test" / "waves" / "VIVOSDEV02" / "VIVOSDEV02_R106.wav"

T_TILE = 4


def _tag_from_prefix(prefix: str) -> str:
    return prefix.strip("/").replace("/", "_").replace(".", "_")


def _bits_from_pow2(pow2_arr: np.ndarray) -> np.ndarray:
    return np.rint(np.log2(pow2_arr)).astype(np.int32)


def pack_all_36_ffn_image(sdram_capacity_bytes: int = 8 * 1024 * 1024) -> tuple[list[int], list[dict]]:
    src_model = onnx.load(str(SRC_FP32_ONNX))
    blocks = discover_ffn_blocks(src_model)

    q_model = onnx.load(str(BITEXACT_W4_ONNX))
    inits = {i.name: numpy_helper.to_array(i) for i in q_model.graph.initializer}

    all_words: list[int] = []
    manifest: list[dict] = []

    raw_wgt_bytes = 0
    raw_param_bytes = 0
    lut_bytes_total = 0

    for idx, blk in enumerate(blocks):
        tag = _tag_from_prefix(blk["prefix"])
        pfx = f"_fpga_{tag}_"

        # ONNX MatMul weights are [K, N]; hardware expects [N, K]
        w1_q = np.rint(inits[pfx + "w1_q"]).astype(np.int8).T.copy()
        b1_q = np.rint(inits[pfx + "b1_q"]).astype(np.int32)
        m1 = np.rint(inits[pfx + "m1"]).astype(np.int32)
        sh1 = _bits_from_pow2(inits[pfx + "pow2_1"])
        lut_f64 = inits[pfx + "lut"]
        lut_i8 = np.roll(lut_f64, -128).astype(np.int8)

        w2_q = np.rint(inits[pfx + "w2_q"]).astype(np.int8).T.copy()
        b2_q = np.rint(inits[pfx + "b2_q"]).astype(np.int32)
        m2 = np.rint(inits[pfx + "m2"]).astype(np.int32)
        sh2 = _bits_from_pow2(inits[pfx + "pow2_2"])

        l1_words, l1_meta = pack_layer_w4_sdram_words(w1_q, b1_q, m1, sh1)
        l2_words, l2_meta = pack_layer_w4_sdram_words(w2_q, b2_q, m2, sh2)

        l1_base = len(all_words)
        all_words.extend(l1_words)
        l2_base = len(all_words)
        all_words.extend(l2_words)

        raw_wgt_bytes += l1_meta["wgt_bytes"] + l2_meta["wgt_bytes"]
        raw_param_bytes += (w1_q.shape[0] + w2_q.shape[0]) * 8
        lut_bytes_total += 256

        manifest.append({
            "block_idx": idx,
            "prefix": blk["prefix"],
            "tag": tag,
            "d_model": int(w1_q.shape[1]),
            "d_ff": int(w1_q.shape[0]),
            "l1_sdram_base_w128": l1_base,
            "l2_sdram_base_w128": l2_base,
            "l1_meta": l1_meta,
            "l2_meta": l2_meta,
            "lut_i8": lut_i8,
            "w1_q": w1_q,
            "b1_q": b1_q,
            "m1": m1,
            "sh1": sh1,
            "w2_q": w2_q,
            "b2_q": b2_q,
            "m2": m2,
            "sh2": sh2,
        })

    packed_bytes = len(all_words) * 16
    assert packed_bytes <= sdram_capacity_bytes, f"Packed image {packed_bytes} B > {sdram_capacity_bytes} B!"

    # Save binary image + JSON layout index (excluding large numpy arrays)
    bin_path = BUILD_DIR / "ffn_30m_sdram_image.bin"
    raw_u8 = bytearray()
    for w in all_words:
        raw_u8.extend(int(w).to_bytes(16, "little"))
    bin_path.write_bytes(bytes(raw_u8))

    index_meta = {
        "total_blocks": len(manifest),
        "raw_w4_weight_bytes": raw_wgt_bytes,
        "raw_64b_param_bytes": raw_param_bytes,
        "raw_lut_bytes": lut_bytes_total,
        "sdram_packed_words_128b": len(all_words),
        "sdram_packed_bytes": packed_bytes,
        "sdram_packed_MiB": round(packed_bytes / (1024.0 * 1024.0), 4),
        "sdram_utilization_pct": round(100.0 * packed_bytes / sdram_capacity_bytes, 2),
        "blocks": [
            {
                "block_idx": m["block_idx"],
                "prefix": m["prefix"],
                "d_model": m["d_model"],
                "d_ff": m["d_ff"],
                "l1_sdram_base_w128": m["l1_sdram_base_w128"],
                "l2_sdram_base_w128": m["l2_sdram_base_w128"],
                "l1_meta": m["l1_meta"],
                "l2_meta": m["l2_meta"],
            }
            for m in manifest
        ],
    }
    (BUILD_DIR / "ffn_30m_sdram_image.json").write_text(json.dumps(index_meta, indent=2), encoding="utf-8")
    return all_words, manifest, index_meta


def extract_real_audio_activations(manifest: list[dict], wav_path: Path) -> list[dict]:
    q_model = onnx.load(str(BITEXACT_W4_ONNX))
    tap_names = []
    for m in manifest:
        pfx = f"_fpga_{m['tag']}_"
        tap_names.extend([pfx + "xq", pfx + "hpost_q", pfx + "yq"])

    existing_outs = {o.name for o in q_model.graph.output}
    for tn in tap_names:
        if tn not in existing_outs:
            q_model.graph.output.append(helper.make_tensor_value_info(tn, TensorProto.DOUBLE, None))

    probe_path = BUILD_DIR / "_g5b_probe.tmp.onnx"
    onnx.save(q_model, str(probe_path))
    sess = ort.InferenceSession(str(probe_path), providers=["CPUExecutionProvider"])
    probe_path.unlink(missing_ok=True)

    audio, sr = sf.read(str(wav_path), dtype="float32")
    feats = compute_fbank(audio, sr)
    outs = sess.run(None, {"x": feats[None], "x_lens": np.array([len(feats)], np.int64)})
    out_map = {o.name: outs[i] for i, o in enumerate(sess.get_outputs())}

    acts = []
    for m in manifest:
        pfx = f"_fpga_{m['tag']}_"
        d_model, d_ff = m["d_model"], m["d_ff"]
        xq_full = np.rint(out_map[pfx + "xq"]).astype(np.int8).reshape(-1, d_model)
        hp_full = np.rint(out_map[pfx + "hpost_q"]).astype(np.int8).reshape(-1, d_ff)
        yq_full = np.rint(out_map[pfx + "yq"]).astype(np.int8).reshape(-1, d_model)
        t_real = xq_full.shape[0]
        assert t_real >= T_TILE, f"Block {m['prefix']} has only T={t_real} < {T_TILE}"

        xq = xq_full[:T_TILE].copy()
        hp_onnx = hp_full[:T_TILE].copy()
        yq_onnx = yq_full[:T_TILE].copy()

        # Verify bit-exact integer equivalence against Python hardware reference BEFORE running FPGA
        acc1 = xq.astype(np.int32) @ m["w1_q"].astype(np.int32).T + m["b1_q"][None, :]
        hpre_py = dyadic_requantize_round_half_up(acc1, m["m1"], m["sh1"])
        hp_py = apply_lut_256(hpre_py, m["lut_i8"])
        acc2 = hp_py.astype(np.int32) @ m["w2_q"].astype(np.int32).T + m["b2_q"][None, :]
        yq_py = dyadic_requantize_round_half_up(acc2, m["m2"], m["sh2"])
        assert np.array_equal(hp_py, hp_onnx), f"Py vs ONNX h_post mismatch on block {m['block_idx']}!"
        assert np.array_equal(yq_py, yq_onnx), f"Py vs ONNX y_out mismatch on block {m['block_idx']}!"

        acts.append({
            "t_real_utt": int(t_real),
            "fbank_frames": int(len(feats)),
            "x_q": xq,
            "hpost_q": hp_onnx,
            "y_q": yq_onnx,
        })
    return acts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    parser.add_argument("--rd-lat", type=int, default=None)
    args = parser.parse_args()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    g4b_path = BUILD_DIR / "g4b_sdram_results.json"
    rd_lat = args.rd_lat
    if rd_lat is None and g4b_path.exists():
        g4b = json.loads(g4b_path.read_text(encoding="utf-8"))
        for key in ("sdram_100mhz", "sdram_80mhz"):
            if key in g4b:
                rd_lat = int(g4b[key]["selected_rd_lat"])
                break
    if rd_lat is None:
        rd_lat = 4

    wav_path = args.wav
    if not wav_path.exists():
        wavs = sorted((ROOT_DIR / "data" / "vivos" / "test" / "waves").rglob("*.wav"))
        wav_path = wavs[0]

    print(f"[G5b] Packing 36 FFN blocks from {BITEXACT_W4_ONNX.name}...")
    all_words, manifest, img_meta = pack_all_36_ffn_image()
    print(
        f"[G5b] Packed {img_meta['total_blocks']} FFN blocks: "
        f"{img_meta['sdram_packed_words_128b']} words = {img_meta['sdram_packed_MiB']} MiB "
        f"({img_meta['sdram_utilization_pct']}% of 8 MiB SDRAM)"
    )

    print(f"[G5b] Extracting real activations & golden tensors on {wav_path.name}...")
    acts = extract_real_audio_activations(manifest, wav_path)
    print(f"[G5b] Extracted & verified Python==ONNX 0-LSB on 36/36 blocks (fbank={acts[0]['fbank_frames']} frames)")

    with UsbBlasterJtag() as dev:
        cap = dev.read_resp(0x2005)
        assert (cap >> 32) & 1 == 1, "SDRAM not ready!"
        eng_mhz, sd_mhz, _, _ = measure_pll_clocks(dev)
        print(f"[G5b] Live FPGA clocks: eng={eng_mhz} MHz, sdram={sd_mhz} MHz, using rd_lat={rd_lat}")

        print("[G5b] Uploading ENTIRE 36-FFN W4A8 model image ONCE into 8 MiB SDRAM...")
        up_s = dev.sdram_write_bulk_words(
            0,
            all_words,
            progress_cb=lambda done, tot: print(
                f"  upload {done}/{tot} ({100.0 * done / tot:.1f}%)", flush=True
            )
            if done % 32768 == 0 or done == tot
            else None,
        )
        print(f"[G5b] Resident model uploaded ONCE in {up_s:.2f} s ({(img_meta['sdram_packed_bytes']/1024.0)/up_s:.2f} KiB/s)")

        block_results = []
        stage_summary: dict[str, dict] = {}
        tot_cyc_all = 0
        mac_cyc_all = 0
        dma_cyc_all = 0
        stall_cyc_all = 0
        act_jtag_bytes_t4 = 0
        act_jtag_s_t4 = 0.0

        for m, a in zip(manifest, acts):
            d_model, d_ff = m["d_model"], m["d_ff"]
            x_words = [int(s, 16) for s in pack_words_128bit_i8(a["x_q"])]
            hpost_golden = [int(s, 16) for s in pack_words_128bit_i8(a["hpost_q"])]
            y_golden = [int(s, 16) for s in pack_words_128bit_i8(a["y_q"])]

            # Upload ONLY T_TILE x d_model input activations + 256B SwooshL LUT (ZERO weights over JTAG!)
            t_io0 = time.perf_counter()
            for idx, w in enumerate(x_words):
                dev.queue_write_128b(OP_COMMIT_ACT, idx, w)
            dev.flush()
            write_lut_256(dev, m["lut_i8"], lut_bank=0)
            up_act_s = time.perf_counter() - t_io0

            # L1: d_model -> d_ff (dst in act_mem at 128)
            c_l1 = run_sdram_stream_layer(
                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                dim_t=T_TILE, k_dim=d_model, n_dim=d_ff, layer_meta=m["l1_meta"],
                sdram_base_w128=m["l1_sdram_base_w128"], rd_lat=rd_lat, lut_mode=1,
                act_base=0, dst_base=128, dst_is_act=1,
            )
            # L2: d_ff -> d_model (read h_post at act_mem 128 -> write out_mem 0)
            c_l2 = run_sdram_stream_layer(
                dev, pack_cfg0_ext, pack_cfg3_ext, pack_cfg1, pack_cfg2,
                dim_t=T_TILE, k_dim=d_ff, n_dim=d_model, layer_meta=m["l2_meta"],
                sdram_base_w128=m["l2_sdram_base_w128"], rd_lat=rd_lat, lut_mode=0,
                act_base=128, dst_base=0, dst_is_act=0,
            )

            t_rd0 = time.perf_counter()
            hpost_hw = [dev.read_128b(1, 128 + i) for i in range(len(hpost_golden))]
            y_hw = [dev.read_128b(0, i) for i in range(len(y_golden))]
            rd_act_s = time.perf_counter() - t_rd0

            hp_mism = sum(1 for x, y in zip(hpost_hw, hpost_golden) if x != y)
            y_mism = sum(1 for x, y in zip(y_hw, y_golden) if x != y)
            assert hp_mism == 0 and y_mism == 0, (
                f"Block {m['block_idx']} ({m['prefix']}) mismatch! hpost={hp_mism}, yout={y_mism}"
            )

            b_tot = c_l1["total_cycles"] + c_l2["total_cycles"]
            b_mac = c_l1["mac_cycles"] + c_l2["mac_cycles"]
            b_dma = c_l1["dma_cycles"] + c_l2["dma_cycles"]
            b_stall = c_l1["stall_cycles"] + c_l2["stall_cycles"]
            b_macs = 2 * T_TILE * d_model * d_ff
            # Host<->FPGA activation traffic: x_in + y_out (h_post stays on-chip in M9K!)
            io_bytes = T_TILE * d_model * 2

            tot_cyc_all += b_tot
            mac_cyc_all += b_mac
            dma_cyc_all += b_dma
            stall_cyc_all += b_stall
            act_jtag_bytes_t4 += io_bytes + 256  # incl. LUT write
            act_jtag_s_t4 += up_act_s + rd_act_s

            parts = m["prefix"].strip("/").split("/")
            stage_id = f"stage_{parts[1]}" if len(parts) >= 2 else "?"
            st_agg = stage_summary.setdefault(stage_id, {
                "stage": stage_id, "d_model": d_model, "blocks": 0,
                "t_utt_frames": a["t_real_utt"], "total_cyc_t4": 0, "mac_cyc_t4": 0, "stall_cyc_t4": 0,
            })
            st_agg["blocks"] += 1
            st_agg["total_cyc_t4"] += b_tot
            st_agg["mac_cyc_t4"] += b_mac
            st_agg["stall_cyc_t4"] += b_stall

            block_results.append({
                "block_idx": m["block_idx"],
                "stage": stage_id,
                "prefix": m["prefix"],
                "shape": f"{d_model}->{d_ff}->{d_model}",
                "total_cycles": b_tot,
                "mac_cycles": b_mac,
                "dma_cycles": b_dma,
                "stall_cycles": b_stall,
                "hw_exec_ms": round(1e3 * b_tot / (eng_mhz * 1e6), 4),
                "macs": b_macs,
                "hpost_mismatch": hp_mism,
                "yout_mismatch": y_mism,
            })
            if m["block_idx"] % 6 == 5 or m["block_idx"] == 0:
                print(
                    f"  [blk {m['block_idx']:02d}/35] stage={stage_id} {d_model}->{d_ff}->{d_model} "
                    f"tot={b_tot} mac={b_mac} stall={b_stall} -> 0 LSB mismatch"
                )

    # Compute exact per-block (T_stage, d_model) on canonical T_fbank=500 (5.0 s audio)
    q_model_500 = onnx.load(str(BITEXACT_W4_ONNX))
    tap_x500 = [f"_fpga_{m['tag']}_xq" for m in manifest]
    existing_outs_500 = {o.name for o in q_model_500.graph.output}
    for tn in tap_x500:
        if tn not in existing_outs_500:
            q_model_500.graph.output.append(helper.make_tensor_value_info(tn, TensorProto.DOUBLE, None))
    p500 = BUILD_DIR / "_g5b_t500_probe.tmp.onnx"
    onnx.save(q_model_500, str(p500))
    sess500 = ort.InferenceSession(str(p500), providers=["CPUExecutionProvider"])
    outs500 = sess500.run(None, {"x": np.zeros((1, 500, 80), np.float32), "x_lens": np.array([500], np.int64)})
    p500.unlink(missing_ok=True)
    sh500 = {o.name: list(r.shape) for o, r in zip(sess500.get_outputs(), outs500)}

    total_ffn_macs_500 = 0
    total_ffn_hw_ms_500 = 0.0
    total_host_act_bytes_500 = 0
    for br, m in zip(block_results, manifest):
        d_m = m["d_model"]
        t_utt_500 = int(sh500[f"_fpga_{m['tag']}_xq"][0])
        n_tiles_t = (t_utt_500 + T_TILE - 1) // T_TILE
        t_last = t_utt_500 - (n_tiles_t - 1) * T_TILE
        exact_macs_blk = 2 * t_utt_500 * d_m * m["d_ff"]
        # Exact cycles for full T=4 tiles + final remainder tile (if t_last != 4)
        frac_cyc = (n_tiles_t - 1 + t_last / float(T_TILE)) * br["total_cycles"]
        total_ffn_macs_500 += exact_macs_blk
        total_ffn_hw_ms_500 += 1e3 * frac_cyc / (eng_mhz * 1e6)
        total_host_act_bytes_500 += t_utt_500 * d_m * 2  # in + out INT8, h_post kept in M9K

    tot_hw_ms_t4 = 1e3 * tot_cyc_all / (eng_mhz * 1e6)
    ref_fbank = acts[0]["fbank_frames"]
    res = {
        "gate": "G5b",
        "device": "EP4CE10E22C8N",
        "sdram_chip": "HY57V641620FTP-H (8 MiB, 16-bit)",
        "wav_file": wav_path.name,
        "fbank_frames": ref_fbank,
        "eng_mhz": eng_mhz,
        "sdram_mhz": sd_mhz,
        "rd_lat": rd_lat,
        "image_meta": img_meta,
        "sdram_upload_once_s": round(up_s, 3),
        "all_36_blocks_zero_lsb_verified": True,
        "t_tile_4_totals": {
            "blocks_verified": len(block_results),
            "layers_verified": len(block_results) * 2,
            "total_cycles": tot_cyc_all,
            "mac_cycles": mac_cyc_all,
            "dma_cycles": dma_cyc_all,
            "stall_cycles": stall_cyc_all,
            "mac_lane_utilization_pct": round(100.0 * mac_cyc_all / tot_cyc_all, 2),
            "stall_pct": round(100.0 * stall_cyc_all / tot_cyc_all, 2),
            "hw_exec_ms": round(tot_hw_ms_t4, 3),
            "host_jtag_act_plus_lut_bytes": act_jtag_bytes_t4,
            "host_jtag_act_io_s": round(act_jtag_s_t4, 3),
        },
        "full_utterance_5s_T500_projection": {
            "ffn_macs_total": total_ffn_macs_500,
            "fpga_ffn_compute_ms": round(total_ffn_hw_ms_500, 2),
            "fpga_ffn_rtf": round(total_ffn_hw_ms_500 / 5000.0, 5),
            "host_fpga_act_bytes_no_weights": total_host_act_bytes_500,
            "host_fpga_act_KiB": round(total_host_act_bytes_500 / 1024.0, 1),
            "note": "Weights (6.19 MiB) remain 100% resident in SDRAM; ONLY activations move over host bus.",
        },
        "stage_summary_t4": list(stage_summary.values()),
        "blocks": block_results,
    }
    out_path = BUILD_DIR / "g5b_30m_sdram_results.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[G5b DONE] 36/36 FFN blocks verified 0 LSB mismatch! Total T=4 compute={tot_hw_ms_t4:.2f} ms. Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
