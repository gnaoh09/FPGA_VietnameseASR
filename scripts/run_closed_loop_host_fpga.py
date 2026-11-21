#!/usr/bin/env python3
"""Q2 Priority #3: True Closed-Loop Host<->FPGA Zipformer-30M Inference Across
Short (1.81s), Medium (3.12s), and Long ~T=500 (4.91s) VIVOS Test Utterances,
explicitly exercising tail tiles with 1, 2, and 3 valid frames (zero-padded to T_TILE=4
and sliced back before feeding downstream host Conv/Attention ONNX layers).

Reports exact byte/tensor equality vs monolithic encoder_fpga_bitexact_w4a8.onnx,
exact transcript equality via stateless RNN-T greedy decoding, and complete audio-to-
transcript wall-clock time breakdown.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import onnx
import onnxruntime as ort
import sherpa_onnx
import soundfile as sf
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fpga_bitexact_30m_onnx import compute_fbank
from eval_30m_wer_pareto_noise import norm_vi
from fpga_jtag_runner import OP_COMMIT_ACT, UsbBlasterJtag
from gen_golden_vectors import pack_words_128bit_i8
from run_full_2448_byte_verify import unpack_words_128bit_i8
from run_g4b_sdram_experiment import measure_pll_clocks
from run_g5b_30m_resident_sdram import BITEXACT_W4_ONNX, pack_all_36_ffn_image
from run_g5b_multisent_fullseq import (
    eval_block_python,
    py_hw_checksum_stream,
    run_layer_fast,
    verify_or_upload_sdram,
)
from sdram_helpers import write_lut_256

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
MODEL_DIR = ROOT_DIR / "data" / "zipformer_30m"
VIVOS_TEST_DIR = ROOT_DIR / "data" / "vivos" / "test"

T_TILE = 4
RD_LAT = 4

TEST_UTTS = [
    {"utt_id": "VIVOSDEV08_274", "length_bucket": "short", "speaker": "VIVOSDEV08"},
    {"utt_id": "VIVOSDEV13_239", "length_bucket": "medium", "speaker": "VIVOSDEV13"},
    {"utt_id": "VIVOSDEV16_182", "length_bucket": "long_near_t500", "speaker": "VIVOSDEV16"},
]

NP_TO_ONNX_DTYPE = {
    np.dtype("float32"): TensorProto.FLOAT,
    np.dtype("float64"): TensorProto.DOUBLE,
    np.dtype("int64"): TensorProto.INT64,
    np.dtype("int32"): TensorProto.INT32,
    np.dtype("int8"): TensorProto.INT8,
    np.dtype("uint8"): TensorProto.UINT8,
    np.dtype("bool"): TensorProto.BOOL,
}


def load_vivos_prompts() -> dict[str, str]:
    prompts = {}
    for line in (VIVOS_TEST_DIR / "prompts.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            uid, txt = line.split(" ", 1)
            prompts[uid] = norm_vi(txt)
    return prompts


def load_tokens() -> dict[int, str]:
    toks = {}
    for line in (MODEL_DIR / "config.json").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            toks[int(parts[1])] = parts[0]
    return toks


def get_node_all_refs(node: onnx.NodeProto) -> list[str]:
    """Collect all tensor/initializer names referenced by a node, including inside If/Loop subgraphs."""
    refs = [inp for inp in node.input if inp]

    def walk_graph(g: onnx.GraphProto) -> None:
        local_init = {i.name for i in g.initializer} | {gi.name for gi in g.input}
        local_prod = {o for sn in g.node for o in sn.output if o}
        for sn in g.node:
            for inp in sn.input:
                if inp and inp not in local_init and inp not in local_prod:
                    refs.append(inp)
            for sa in sn.attribute:
                if sa.HasField("g"):
                    walk_graph(sa.g)
                for sg in sa.graphs:
                    walk_graph(sg)

    for a in node.attribute:
        if a.HasField("g"):
            walk_graph(a.g)
        for g in a.graphs:
            walk_graph(g)
    return refs


def build_37_host_segments(
    onnx_path: Path,
    manifest: list[dict],
) -> tuple[list[dict], ort.InferenceSession]:
    """Split encoder_fpga_bitexact_w4a8.onnx cleanly into 37 segments around the 36 FFN blocks."""
    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    init_map = {i.name: i for i in model.graph.initializer}
    nodes = list(model.graph.node)

    edge_dtype: dict[str, int] = {
        v.name: int(v.type.tensor_type.elem_type)
        for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    }

    # Locate xq node index and yq node index for each of 36 blocks in topological order
    out_to_idx = {}
    for idx, n in enumerate(nodes):
        for o in n.output:
            if o:
                out_to_idx[o] = idx

    ranges = []
    prev_start = 0
    for bi, m in enumerate(manifest):
        pfx = f"_fpga_{m['tag']}_"
        xq_name = pfx + "xq"
        yq_name = pfx + "yq"
        xq_idx = out_to_idx[xq_name]
        yq_idx = out_to_idx[yq_name]
        assert prev_start <= xq_idx < yq_idx
        seg_nodes = nodes[prev_start : xq_idx + 1]
        ranges.append((bi, seg_nodes, xq_name, yq_name))
        # Skip hardware FFN nodes (xq_idx+1 .. yq_idx inclusive); next segment resumes at yq_idx+1
        prev_start = yq_idx + 1
    # Final segment (seg_36) outputs encoder_out & encoder_out_lens
    ranges.append((36, nodes[prev_start:], None, None))

    # Determine which produced outputs of each segment are needed by any downstream segment or final out
    final_outs = {o.name for o in model.graph.output}
    needed_after: list[set[str]] = [set() for _ in range(37)]
    running_reqs = set(final_outs)
    for k in range(36, -1, -1):
        needed_after[k] = set(running_reqs)
        _bi, s_nodes, _xq, yq_n = ranges[k]
        for n in s_nodes:
            for inp in get_node_all_refs(n):
                if inp and inp not in init_map:
                    running_reqs.add(inp)
        if yq_n:
            running_reqs.add(yq_n)

    segments = []
    tmp_dir = BUILD_DIR / "_closed_loop_segs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for k in range(37):
        bi, s_nodes, xq_name, _yq_n = ranges[k]
        produced = {o for n in s_nodes for o in n.output if o}
        consumed = []
        seen_c = set()
        for n in s_nodes:
            for inp in get_node_all_refs(n):
                if inp and inp not in init_map and inp not in produced and inp not in seen_c:
                    seen_c.add(inp)
                    consumed.append(inp)

        out_names = []
        seen_o = set()
        for o in produced:
            if o in needed_after[k] and o not in seen_o:
                seen_o.add(o)
                out_names.append(o)
        if xq_name and xq_name not in seen_o:
            out_names.append(xq_name)

        used_inits = {inp for n in s_nodes for inp in get_node_all_refs(n) if inp in init_map}
        sub_inits = [init_map[name] for name in used_inits]

        sub_inputs = [
            helper.make_tensor_value_info(name, edge_dtype[name], None) for name in consumed
        ]
        sub_outputs = [
            helper.make_tensor_value_info(name, edge_dtype[name], None) for name in out_names
        ]
        sub_graph = helper.make_graph(s_nodes, f"zipformer_seg_{k}", sub_inputs, sub_outputs, sub_inits)
        sub_model = helper.make_model(sub_graph, opset_imports=model.opset_import)
        sub_model.ir_version = min(model.ir_version, 9)
        seg_path = tmp_dir / f"seg_{k:02d}.onnx"
        onnx.save(sub_model, str(seg_path))
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(seg_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        segments.append({
            "seg_idx": k,
            "block_idx": bi if bi < 36 else None,
            "inputs": consumed,
            "outputs": out_names,
            "xq_name": xq_name,
            "yq_name": (f"_fpga_{manifest[bi]['tag']}_yq" if bi < 36 else None),
            "sess": sess,
        })

    # Monolithic baseline session for exact verification
    mono_opts = ort.SessionOptions()
    mono_opts.intra_op_num_threads = 1
    mono_sess = ort.InferenceSession(str(onnx_path), sess_options=mono_opts, providers=["CPUExecutionProvider"])
    return segments, mono_sess


def greedy_rnnt_decode(
    enc_out: np.ndarray,
    enc_lens: np.ndarray,
    dec_sess: ort.InferenceSession,
    jn_sess: ort.InferenceSession,
    tokens: dict[int, str],
) -> str:
    """Stateless Zipformer greedy search decoding matching sherpa_onnx."""
    T = int(enc_lens[0])
    hyp = [0, 0]
    dec_out = dec_sess.run(None, {"y": np.array([hyp], np.int64)})[0]
    for t in range(T):
        logit = jn_sess.run(None, {"encoder_out": enc_out[:, t, :], "decoder_out": dec_out})[0][0]
        y = int(np.argmax(logit))
        if y != 0:
            hyp.append(y)
            dec_out = dec_sess.run(None, {"y": np.array([hyp[-2:]], np.int64)})[0]
    return "".join(tokens[i] for i in hyp[2:]).replace("\u2581", " ").strip()


def run_single_utterance_closed_loop(
    wav_path: Path,
    segments: list[dict],
    manifest: list[dict],
    dev: UsbBlasterJtag | None,
    dec_sess: ort.InferenceSession,
    jn_sess: ort.InferenceSession,
    tokens: dict[int, str],
    mono_sess: ort.InferenceSession | None,
    ref_text: str,
    verify_golden: bool = True,
) -> dict:
    """Execute one utterance through 37 ONNX host segments + 36 physical FPGA FFN blocks."""
    t_wall0 = time.perf_counter()

    t_fbank0 = time.perf_counter()
    audio, sr = sf.read(str(wav_path), dtype="float32")
    dur_s = len(audio) / float(sr)
    feats = compute_fbank(audio, sr)
    t_host_s = time.perf_counter() - t_fbank0

    env: dict[str, np.ndarray] = {
        "x": feats[None],
        "x_lens": np.array([len(feats)], np.int64),
    }

    t_fpga_compute_s = 0.0
    t_jtag_write_s = 0.0
    t_jtag_read_s = 0.0
    t_ctrl_lut_sync_s = 0.0

    total_tiles = 0
    padded_tail_tiles = 0
    csum_mism = 0
    byte_mism = 0
    max_abs_yq_diff = 0
    act_in_bytes = 0
    act_out_bytes = 0
    tot_cycles = 0
    mac_cycles = 0
    stall_cycles = 0
    stage_tiles: dict[str, dict] = {}

    for seg in segments:
        t_h0 = time.perf_counter()
        feed = {k: env[k] for k in seg["inputs"]}
        outs = seg["sess"].run(seg["outputs"], feed)
        for name, val in zip(seg["outputs"], outs):
            env[name] = val
        t_host_s += time.perf_counter() - t_h0

        bi = seg["block_idx"]
        if bi is None:
            continue

        m = manifest[bi]
        d_m, d_ff = m["d_model"], m["d_ff"]
        n_tiles_l2 = m["l2_meta"]["n_tiles"]
        stage_id = f"stage_{m['prefix'].strip('/').split('/')[1]}"

        xq_orig = env[seg["xq_name"]]  # [T_real, 1, d_m] float64 exact integers -128..127
        t_real = int(xq_orig.shape[0])
        xq_i8 = np.rint(xq_orig).astype(np.int8).reshape(t_real, d_m)
        rem = t_real % T_TILE
        pad = (T_TILE - rem) if rem != 0 else 0
        if pad:
            padded_tail_tiles += 1
            xq_padded = np.vstack([xq_i8, np.zeros((pad, d_m), dtype=np.int8)])
        else:
            xq_padded = xq_i8
        n_t_tiles = xq_padded.shape[0] // T_TILE

        if stage_id not in stage_tiles:
            stage_tiles[stage_id] = {"t_frames": t_real, "mod4": rem, "padded_tail": pad > 0, "n_t_tiles": n_t_tiles}

        # Exact integer golden for tile verification (skipped in --no-verify pure inference mode)
        yq_gold_full = None
        if verify_golden or dev is None:
            _, _, yq_gold_full, _, _, _, _ = eval_block_python(m, xq_padded)
        yq_hw_full = np.empty_like(xq_padded, dtype=np.int8)

        if dev is not None:
            t_c0 = time.perf_counter()
            write_lut_256(dev, m["lut_i8"], lut_bank=0)
            t_ctrl_lut_sync_s += time.perf_counter() - t_c0

        for ti in range(n_t_tiles):
            s0 = ti * T_TILE
            xq_t = xq_padded[s0 : s0 + T_TILE]
            act_in_bytes += T_TILE * d_m
            act_out_bytes += T_TILE * d_m
            total_tiles += 1

            if dev is None:
                assert yq_gold_full is not None
                yq_hw_full[s0 : s0 + T_TILE] = yq_gold_full[s0 : s0 + T_TILE]
                continue

            t_w0 = time.perf_counter()
            for idx_w, w in enumerate(int(s, 16) for s in pack_words_128bit_i8(xq_t)):
                dev.queue_write_128b(OP_COMMIT_ACT, idx_w, w)
            dev.flush()
            t_jtag_write_s += time.perf_counter() - t_w0

            # Measure Layer 1 & Layer 2 launch+wait; separate pure hardware cycles vs JTAG poll/ctrl
            t_run0 = time.perf_counter()
            c_l1 = run_layer_fast(dev, T_TILE, d_m, d_ff, m["l1_meta"], m["l1_sdram_base_w128"], RD_LAT, 1, 0, 128, 1)
            c_l2 = run_layer_fast(dev, T_TILE, d_ff, d_m, m["l2_meta"], m["l2_sdram_base_w128"], RD_LAT, 0, 128, 0, 0)
            dt_run = time.perf_counter() - t_run0

            cyc_blk = c_l1["total_cycles"] + c_l2["total_cycles"]
            tot_cycles += cyc_blk
            mac_cycles += c_l1["mac_cycles"] + c_l2["mac_cycles"]
            stall_cycles += c_l1["stall_cycles"] + c_l2["stall_cycles"]
            # Attribute exact FPGA clock duration to fpga_compute and residual polling/regs to ctrl_sync
            fpga_s = cyc_blk / (80.0e6)  # refined below using measured eng_mhz if needed
            t_fpga_compute_s += fpga_s
            t_ctrl_lut_sync_s += max(0.0, dt_run - fpga_s)

            # Mandatory closed-loop JTAG readback: host segment k+1 depends on FPGA y_out
            t_r0 = time.perf_counter()
            hw_words = [dev.read_128b(0, i) for i in range((d_m // 16) * T_TILE)]
            t_jtag_read_s += time.perf_counter() - t_r0

            yq_t_hw = unpack_words_128bit_i8(hw_words, T_TILE, d_m)
            if verify_golden:
                assert yq_gold_full is not None
                yq_t_gold = yq_gold_full[s0 : s0 + T_TILE]
                if int(c_l2["out_checksum"], 16) != py_hw_checksum_stream(yq_t_gold, n_tiles_l2):
                    csum_mism += 1
                    raise AssertionError(f"Checksum mismatch {wav_path.name} blk={bi} tile={ti}")
                bm = int(np.sum(yq_t_hw.view(np.uint8) != yq_t_gold.view(np.uint8)))
                mad = int(np.max(np.abs(yq_t_hw.astype(np.int16) - yq_t_gold.astype(np.int16))))
                byte_mism += bm
                max_abs_yq_diff = max(max_abs_yq_diff, mad)
                assert bm == 0 and mad == 0, f"Byte mismatch {wav_path.name} blk={bi} tile={ti}"
            yq_hw_full[s0 : s0 + T_TILE] = yq_t_hw

        # Crucial closed-loop step: slice away tail zero padding and feed TRUE FPGA y_out back into host ONNX!
        t_h1 = time.perf_counter()
        env[seg["yq_name"]] = yq_hw_full[:t_real].reshape(xq_orig.shape).astype(np.float64)
        t_host_s += time.perf_counter() - t_h1

    t_dec0 = time.perf_counter()
    fpga_transcript = greedy_rnnt_decode(env["encoder_out"], env["encoder_out_lens"], dec_sess, jn_sess, tokens)
    t_host_s += time.perf_counter() - t_dec0
    t_wall_total_s = time.perf_counter() - t_wall0

    # Compare closed-loop encoder_out and transcript against monolithic encoder_fpga_bitexact_w4a8.onnx
    enc_max_diff = None
    transcript_match = None
    if mono_sess is not None:
        mono_out, mono_lens = mono_sess.run(None, {"x": feats[None], "x_lens": np.array([len(feats)], np.int64)})
        enc_max_diff = float(np.max(np.abs(env["encoder_out"] - mono_out)))
        mono_transcript = greedy_rnnt_decode(mono_out, mono_lens, dec_sess, jn_sess, tokens)
        transcript_match = (fpga_transcript == mono_transcript)
        if verify_golden:
            assert enc_max_diff == 0.0, f"Closed-loop encoder_out differed from ONNX: {enc_max_diff}"
            assert transcript_match, f"Transcript diff: {fpga_transcript!r} vs {mono_transcript!r}"

    wer_wo = jiwer.process_words(ref_text, norm_vi(fpga_transcript))
    utt_wer_pct = round(100.0 * (wer_wo.substitutions + wer_wo.deletions + wer_wo.insertions) / max(1, wer_wo.hits + wer_wo.substitutions + wer_wo.deletions), 2)

    return {
        "utt_id": wav_path.stem,
        "verify_golden": verify_golden,
        "duration_s": round(dur_s, 3),
        "fbank_frames": int(len(feats)),
        "stage_shapes": stage_tiles,
        "total_block_tiles": total_tiles,
        "padded_tail_block_launches": padded_tail_tiles,
        "csum_mismatches": csum_mism if verify_golden else None,
        "y_out_byte_mismatches": byte_mism if verify_golden else None,
        "y_out_max_abs_diff_lsb": max_abs_yq_diff if verify_golden else None,
        "encoder_out_max_abs_diff": enc_max_diff,
        "transcript_exact_match_vs_onnx": transcript_match,
        "ref_text": ref_text,
        "fpga_transcript": norm_vi(fpga_transcript),
        "utt_wer_pct": utt_wer_pct,
        "act_in_bytes_padded": act_in_bytes,
        "act_out_bytes_padded": act_out_bytes,
        "hw_total_cycles": tot_cycles,
        "hw_mac_cycles": mac_cycles,
        "hw_stall_cycles": stall_cycles,
        "timing_breakdown_s": {
            "t_fpga_compute_s": round(t_fpga_compute_s, 4),
            "t_jtag_act_write_s": round(t_jtag_write_s, 3),
            "t_jtag_act_read_s": round(t_jtag_read_s, 3),
            "t_ctrl_lut_sync_s": round(t_ctrl_lut_sync_s, 3),
            "t_host_fbank_nonffn_decode_s": round(t_host_s, 3),
            "t_wall_audio_to_transcript_s": round(t_wall_total_s, 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Verify 37-segment graph & transcripts without FPGA")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Pure inference mode: skip host golden FFN & byte-diff checks while retaining mandatory JTAG y_out readback",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of passes per selected utterance (reports median & min-max)")
    parser.add_argument("--utt-id", type=str, default=None, help="Optional single utterance ID filter (e.g. VIVOSDEV16_182)")
    args = parser.parse_args()

    verify_golden = not args.no_verify
    selected_utts = [u for u in TEST_UTTS if args.utt_id is None or u["utt_id"] == args.utt_id]
    if not selected_utts:
        raise ValueError(f"Unknown --utt-id {args.utt_id!r}")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_vivos_prompts()
    tokens = load_tokens()
    all_words, manifest, _ = pack_all_36_ffn_image()
    segments, mono_sess = build_37_host_segments(BITEXACT_W4_ONNX, manifest)

    dec_opts = ort.SessionOptions()
    dec_opts.intra_op_num_threads = 1
    jn_opts = ort.SessionOptions()
    jn_opts.intra_op_num_threads = 1
    dec_sess = ort.InferenceSession(str(MODEL_DIR / "decoder-epoch-20-avg-10.onnx"), sess_options=dec_opts, providers=["CPUExecutionProvider"])
    jn_sess = ort.InferenceSession(str(MODEL_DIR / "joiner-epoch-20-avg-10.onnx"), sess_options=jn_opts, providers=["CPUExecutionProvider"])

    if args.dry_run:
        for spec in selected_utts:
            wp = VIVOS_TEST_DIR / "waves" / spec["speaker"] / f"{spec['utt_id']}.wav"
            res = run_single_utterance_closed_loop(
                wp, segments, manifest, None, dec_sess, jn_sess, tokens, mono_sess,
                prompts[spec["utt_id"]], verify_golden=verify_golden,
            )
            print(
                f"[DRY-RUN PASS verify={verify_golden}] {spec['utt_id']} T={res['fbank_frames']} "
                f"tiles={res['total_block_tiles']} enc_diff={res['encoder_out_max_abs_diff']} txt={res['fpga_transcript']}"
            )
        return 0

    results = []
    with UsbBlasterJtag() as dev:
        cap = dev.read_resp(0x2005)
        assert (cap >> 32) & 1 == 1, "SDRAM not ready!"
        eng_mhz, sd_mhz, _, _ = measure_pll_clocks(dev)
        already_ok, up_s = verify_or_upload_sdram(dev, all_words, manifest)
        print(f"[Closed-Loop verify={verify_golden}] eng={eng_mhz} MHz, sdram={sd_mhz} MHz | resident_reused={already_ok} (upload={up_s:.1f}s)")

        for spec in selected_utts:
            wp = VIVOS_TEST_DIR / "waves" / spec["speaker"] / f"{spec['utt_id']}.wav"
            for rep_idx in range(args.repeat):
                res = run_single_utterance_closed_loop(
                    wp, segments, manifest, dev, dec_sess, jn_sess, tokens, mono_sess,
                    prompts[spec["utt_id"]], verify_golden=verify_golden,
                )
                exact_fpga_s = res["hw_total_cycles"] / (eng_mhz * 1e6)
                tb = res["timing_breakdown_s"]
                diff_eng = exact_fpga_s - tb["t_fpga_compute_s"]
                tb["t_fpga_compute_s"] = round(exact_fpga_s, 4)
                tb["t_ctrl_lut_sync_s"] = round(max(0.0, tb["t_ctrl_lut_sync_s"] - diff_eng), 3)
                res["repeat_idx"] = rep_idx
                res["length_bucket"] = spec["length_bucket"]
                res["measured_rtf_ffn"] = round(exact_fpga_s / res["duration_s"], 5)
                res["measured_rtf_wall"] = round(tb["t_wall_audio_to_transcript_s"] / res["duration_s"], 2)
                res["mac_lane_util_pct"] = round(100.0 * res["hw_mac_cycles"] / res["hw_total_cycles"], 2)
                res["stall_pct"] = round(100.0 * res["hw_stall_cycles"] / res["hw_total_cycles"], 2)
                results.append(res)
                print(
                    f"[CLOSED-LOOP PASS rep={rep_idx+1}/{args.repeat} verify={verify_golden}] {spec['utt_id']} "
                    f"({spec['length_bucket']}, {res['duration_s']}s, T={res['fbank_frames']}): "
                    f"fpga={exact_fpga_s*1e3:.1f}ms (RTF_FFN={res['measured_rtf_ffn']}) | "
                    f"wall={tb['t_wall_audio_to_transcript_s']}s | transcript_match={res['transcript_exact_match_vs_onnx']}"
                )

    repeat_summary = {}
    for spec in selected_utts:
        uid = spec["utt_id"]
        sub = [r for r in results if r["utt_id"] == uid]
        walls = [r["timing_breakdown_s"]["t_wall_audio_to_transcript_s"] for r in sub]
        repeat_summary[uid] = {
            "runs": len(sub),
            "wall_s_median": round(float(np.median(walls)), 3),
            "wall_s_min": round(float(min(walls)), 3),
            "wall_s_max": round(float(max(walls)), 3),
        }

    out = {
        "experiment": "Q2_Priority_3_Closed_Loop_Host_FPGA_Multi_Length",
        "device": "EP4CE10E22C8N",
        "verify_golden": verify_golden,
        "repeat": args.repeat,
        "eng_mhz": eng_mhz,
        "sdram_mhz": sd_mhz,
        "sdram_image_reused": already_ok,
        "repeat_summary": repeat_summary,
        "utterances": results,
    }
    out_name = "closed_loop_host_fpga_results.json" if (verify_golden and args.repeat == 1 and args.utt_id is None) else "closed_loop_pure_inference_repeat.json"
    out_path = BUILD_DIR / out_name
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved closed-loop results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
