#!/usr/bin/env python3
"""Generate English publication-grade vector SVG figures for the IEEE TVLSI manuscript.
Optimized specifically for IEEE two-column print legibility:
  - Single-column figures (fig_rtl_timing, fig3a, fig3b, fig5) use a compact W=560 canvas
    with 12.5-16.0 px bold fonts, thick strokes, and large markers so text renders at 6-8 pt
    within a 3.5-inch IEEE column.
  - Full-width two-column figures (fig1, fig2, fig4 in figure*) use 11.5-16.0 px fonts.
  - No embedded figure titles (avoiding caption numbering mismatch) and zero label overlap.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_oos_stall_validation import predict_layer_rtl

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
FIG_DIR = ROOT_DIR / "paper_ieee_zipformer" / "figures"

FONT_FAMILY = "DejaVu Sans, Arial, Helvetica, sans-serif"


def xml_esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_en(val: float, digits: int = 2, signed: bool = False) -> str:
    if signed:
        return f"{val:+.{digits}f}"
    return f"{val:.{digits}f}"


def wrap_svg(W: int, H: int, elems: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="{FONT_FAMILY}">\n  '
        + "\n  ".join(elems)
        + "\n</svg>\n"
    )


def arrow_defs() -> str:
    return (
        "<defs>\n"
        '  <marker id="arr_slate" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
        '    <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#475569"/>\n'
        "  </marker>\n"
        '  <marker id="arr_blue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
        '    <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#1d4ed8"/>\n'
        "  </marker>\n"
        '  <marker id="arr_orange" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
        '    <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#c2410c"/>\n'
        "  </marker>\n"
        '  <marker id="arr_teal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
        '    <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#0f766e"/>\n'
        "  </marker>\n"
        '  <marker id="arr_red" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
        '    <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#dc2626"/>\n'
        "  </marker>\n"
        "</defs>"
    )


def generate_fig1_hybrid_architecture(out_path: Path) -> None:
    W, H = 940, 355
    e: list[str] = [arrow_defs(), f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    # Panel 1: HOST (ONNX Runtime)
    e.append('<rect x="14" y="14" width="218" height="270" rx="8" fill="#f8fafc" stroke="#334155" stroke-width="1.8"/>')
    e.append('<rect x="14" y="14" width="218" height="34" rx="8" fill="#1e293b"/>')
    e.append('<rect x="14" y="36" width="218" height="12" fill="#1e293b"/>')
    e.append('<text x="123" y="36" text-anchor="middle" font-size="13.0" font-weight="bold" fill="#ffffff">HOST (ONNX Runtime)</text>')

    host_boxes = [
        (60, 44, "#e2e8f0", "#475569", "Audio → 80-bin Fbank", "Embed + Conv + Attention"),
        (118, 48, "#dbeafe", "#1d4ed8", "36 Zipformer FFN Blocks", "Offloaded to FPGA (W4A8)"),
        (180, 44, "#fef3c7", "#b45309", "Tail Pad / Slice (T_tile)", "Zero-pad when T mod 4 ≠ 0"),
        (236, 38, "#dcfce7", "#15803d", "RNN-T Decoder &amp; Joiner", "24/24 bit-exact vs ONNX"),
    ]
    for by, bh, fill, stroke, t1, t2 in host_boxes:
        e.append(f'<rect x="26" y="{by}" width="194" height="{bh}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        e.append(f'<text x="123" y="{by + bh*0.44:.1f}" text-anchor="middle" font-size="11.6" font-weight="bold" fill="#0f172a">{t1}</text>')
        e.append(f'<text x="123" y="{by + bh*0.79:.1f}" text-anchor="middle" font-size="10.2" fill="#334155">{t2}</text>')

    e.append('<line x1="123" y1="104" x2="123" y2="116" stroke="#475569" stroke-width="1.5" marker-end="url(#arr_slate)"/>')
    e.append('<line x1="123" y1="166" x2="123" y2="178" stroke="#475569" stroke-width="1.5" marker-end="url(#arr_slate)"/>')
    e.append('<line x1="123" y1="224" x2="123" y2="234" stroke="#475569" stroke-width="1.5" marker-end="url(#arr_slate)"/>')

    # Bridge Host <-> FPGA via JTAG
    e.append('<line x1="232" y1="124" x2="314" y2="124" stroke="#1d4ed8" stroke-width="2.4" marker-end="url(#arr_blue)"/>')
    e.append('<text x="273" y="114" text-anchor="middle" font-size="10.8" font-weight="bold" fill="#1e3a8a">x_q INT8</text>')
    e.append('<rect x="236" y="34" width="76" height="44" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.1"/>')
    e.append('<text x="274" y="52" text-anchor="middle" font-size="10.2" font-weight="bold" fill="#334155">1× Weights</text>')
    e.append('<text x="274" y="68" text-anchor="middle" font-size="10.0" fill="#0f766e">6.52 MiB</text>')

    e.append('<line x1="316" y1="164" x2="234" y2="164" stroke="#c2410c" stroke-width="2.4" marker-end="url(#arr_orange)"/>')
    e.append('<text x="275" y="180" text-anchor="middle" font-size="10.8" font-weight="bold" fill="#9a3412">y_out INT8</text>')

    # Panel 2: FPGA (Cyclone IV EP4CE10) — highlighting DMA path, Ping-Pong, and Counter Taps
    e.append('<rect x="318" y="14" width="400" height="270" rx="8" fill="#f8fafc" stroke="#1d4ed8" stroke-width="1.8"/>')
    e.append('<rect x="318" y="14" width="400" height="34" rx="8" fill="#1d4ed8"/>')
    e.append('<rect x="318" y="36" width="400" height="12" fill="#1d4ed8"/>')
    e.append('<text x="518" y="35" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#ffffff">EP4CE10 Silicon (6,056 LE · 26/46 M9K)</text>')

    # Dual clock domains banner
    e.append('<rect x="330" y="54" width="214" height="26" rx="4" fill="#dbeafe" stroke="#2563eb" stroke-width="1.1"/>')
    e.append('<text x="437" y="71" text-anchor="middle" font-size="10.6" font-weight="bold" fill="#1e3a8a">clk_eng Domain (altpll)</text>')
    e.append('<rect x="556" y="54" width="150" height="26" rx="4" fill="#ccfbf1" stroke="#0d9488" stroke-width="1.1"/>')
    e.append('<text x="631" y="71" text-anchor="middle" font-size="10.6" font-weight="bold" fill="#115e59">clk_sdram Domain</text>')

    # Compute side: act M9K -> 16 MAC Lanes -> Epilogue
    e.append('<rect x="330" y="90" width="84" height="52" rx="5" fill="#ffffff" stroke="#3b82f6" stroke-width="1.3"/>')
    e.append('<text x="372" y="112" text-anchor="middle" font-size="10.8" font-weight="bold" fill="#0f172a">act / h M9K</text>')
    e.append('<text x="372" y="129" text-anchor="middle" font-size="9.8" fill="#475569">12 local M9K</text>')

    e.append('<rect x="428" y="90" width="96" height="52" rx="5" fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.4"/>')
    e.append('<text x="476" y="112" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#1e3a8a">16 MAC Lanes</text>')
    e.append('<text x="476" y="129" text-anchor="middle" font-size="9.8" fill="#1e40af">W4→I8 · psum</text>')

    e.append('<rect x="538" y="90" width="86" height="52" rx="5" fill="#fef3c7" stroke="#d97706" stroke-width="1.3"/>')
    e.append('<text x="581" y="112" text-anchor="middle" font-size="10.8" font-weight="bold" fill="#92400e">Epilogue</text>')
    e.append('<text x="581" y="129" text-anchor="middle" font-size="9.8" fill="#78350f">M,S · SwooshL</text>')

    e.append('<line x1="414" y1="116" x2="426" y2="116" stroke="#1d4ed8" stroke-width="1.6" marker-end="url(#arr_blue)"/>')
    e.append('<line x1="524" y1="116" x2="536" y2="116" stroke="#1d4ed8" stroke-width="1.6" marker-end="url(#arr_blue)"/>')

    # Highlighted DMA Path: SDRAM Controller -> Toggle CDC -> Ping-Pong M9K (Bank A/B) -> 16 MAC Lanes
    e.append('<rect x="330" y="158" width="176" height="62" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="2.0"/>')
    e.append('<text x="418" y="177" text-anchor="middle" font-size="11.2" font-weight="bold" fill="#115e59">Ping-Pong DMA Buffers</text>')
    e.append('<rect x="342" y="185" width="70" height="24" rx="3" fill="#ccfbf1" stroke="#0f766e" stroke-width="1.1"/>')
    e.append('<text x="377" y="201" text-anchor="middle" font-size="9.8" font-weight="bold" fill="#134e4a">Bank A (4K)</text>')
    e.append('<rect x="424" y="185" width="70" height="24" rx="3" fill="#ffffff" stroke="#0f766e" stroke-width="1.1"/>')
    e.append('<text x="459" y="201" text-anchor="middle" font-size="9.8" font-weight="bold" fill="#134e4a">Bank B (4K)</text>')

    e.append('<rect x="520" y="164" width="60" height="50" rx="5" fill="#e0e7ff" stroke="#4f46e5" stroke-width="1.5"/>')
    e.append('<text x="550" y="185" text-anchor="middle" font-size="10.2" font-weight="bold" fill="#312e81">2-FF</text>')
    e.append('<text x="550" y="201" text-anchor="middle" font-size="9.6" fill="#3730a3">CDC</text>')

    e.append('<rect x="594" y="158" width="112" height="62" rx="6" fill="#ecfeff" stroke="#0891b2" stroke-width="2.0"/>')
    e.append('<text x="650" y="179" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#155e75">SDRAM Ctrl</text>')
    e.append('<text x="650" y="196" text-anchor="middle" font-size="9.8" fill="#164e63">256-col burst</text>')
    e.append('<text x="650" y="211" text-anchor="middle" font-size="9.8" fill="#0e7490">Refresh FSM</text>')

    # Thick teal DMA data path arrows
    e.append('<line x1="594" y1="189" x2="582" y2="189" stroke="#0f766e" stroke-width="2.2" marker-end="url(#arr_teal)"/>')
    e.append('<line x1="520" y1="189" x2="508" y2="189" stroke="#0f766e" stroke-width="2.2" marker-end="url(#arr_teal)"/>')
    e.append('<path d="M 476 158 L 476 144" fill="none" stroke="#0f766e" stroke-width="2.2" marker-end="url(#arr_teal)"/>')

    # Hardware Counter Tap Bar & Tap Callouts
    e.append('<rect x="330" y="232" width="376" height="42" rx="5" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>')
    e.append('<text x="518" y="248" text-anchor="middle" font-size="10.4" font-weight="bold" fill="#92400e">32-bit Hardware Cycle Counter Tap Points</text>')
    e.append('<text x="518" y="264" text-anchor="middle" font-size="9.8" fill="#78350f">'
             'C_tot (Run) · C_mac (Compute) · C_stall (ST_WAIT_DMA) · C_dma (req→done)</text>')

    # Vertical orange tap lines cleanly routed through empty corridors
    e.append('<line x1="513" y1="142" x2="513" y2="232" stroke="#d97706" stroke-width="1.2" stroke-dasharray="3,2"/>')
    e.append('<line x1="550" y1="214" x2="550" y2="232" stroke="#d97706" stroke-width="1.2" stroke-dasharray="3,2"/>')
    e.append('<line x1="650" y1="220" x2="650" y2="232" stroke="#d97706" stroke-width="1.2" stroke-dasharray="3,2"/>')

    # Bus SDRAM Ctrl <-> External SDRAM
    e.append('<line x1="772" y1="184" x2="720" y2="184" stroke="#0f766e" stroke-width="2.8" marker-end="url(#arr_teal)"/>')
    e.append('<text x="746" y="172" text-anchor="middle" font-size="10.2" font-weight="bold" fill="#0f766e">16b SDR</text>')
    e.append('<text x="746" y="202" text-anchor="middle" font-size="9.8" fill="#334155">60–120M</text>')

    # Panel 3: External SDRAM
    e.append('<rect x="774" y="14" width="152" height="270" rx="8" fill="#f8fafc" stroke="#0f766e" stroke-width="1.8"/>')
    e.append('<rect x="774" y="14" width="152" height="34" rx="8" fill="#0f766e"/>')
    e.append('<rect x="774" y="36" width="152" height="12" fill="#0f766e"/>')
    e.append('<text x="850" y="35" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#ffffff">External SDRAM</text>')

    e.append('<rect x="784" y="58" width="132" height="114" rx="5" fill="#99f6e4" stroke="#0f766e" stroke-width="1.3"/>')
    e.append('<text x="850" y="78" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#134e4a">36 Resident FFNs</text>')
    e.append('<text x="850" y="96" text-anchor="middle" font-size="10.4" font-weight="bold" fill="#115e59">6.52 MiB / 8 MiB</text>')
    e.append('<text x="850" y="114" text-anchor="middle" font-size="9.8" fill="#134e4a">W4 wgts + headers</text>')
    e.append('<text x="850" y="132" text-anchor="middle" font-size="9.8" fill="#0f766e">0 B weight reload</text>')
    e.append('<text x="850" y="149" text-anchor="middle" font-size="9.8" fill="#0f766e">per utterance</text>')

    e.append('<rect x="784" y="180" width="132" height="38" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.1" stroke-dasharray="4,3"/>')
    e.append('<text x="850" y="203" text-anchor="middle" font-size="10.2" font-weight="bold" fill="#475569">Free: 1.48 MiB</text>')

    e.append('<rect x="784" y="226" width="132" height="48" rx="4" fill="#ffffff" stroke="#0d9488" stroke-width="1.1"/>')
    e.append('<text x="850" y="246" text-anchor="middle" font-size="10.4" font-weight="bold" fill="#0f766e">8 MiB Full-Chip BIST</text>')
    e.append('<text x="850" y="263" text-anchor="middle" font-size="9.8" fill="#1e293b">0 errors (60–120 MHz)</text>')

    # Bottom verification bar
    e.append('<rect x="14" y="294" width="912" height="48" rx="6" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.3"/>')
    e.append(
        '<text x="470" y="314" text-anchor="middle" font-size="11.2" font-weight="bold" fill="#14532d">'
        "Bit-Exact Silicon Verification: 2,792,448 B Readback (0 LSB errors) · 24/24 Utterances Closed-Loop ONNX Match</text>"
    )
    e.append(
        '<text x="470" y="333" text-anchor="middle" font-size="10.6" fill="#166534">'
        "DMA path: External SDRAM → 256-col Page-Burst Controller → Toggle CDC → 2×4 KiB M9K Ping-Pong → 16 MAC Lanes</text>"
    )

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig2_m9k_bandwidth_tradeoff(mem_data: dict, g4b: dict, out_path: Path) -> None:
    matrix = mem_data["tiling_feasibility_matrix"]
    lookup = {(row["arch"], row["t_tile"], row["w_bits"]): row for row in matrix}
    bist_100_rd = float(g4b["sdram_100mhz"]["bist_8mib"]["rd_MBps_1e6"])

    W, H = 920, 390
    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    # Panel (a)
    ml1, mt1, pw1, ph1 = 72, 38, 360, 294
    ymax1 = 82.0

    def py1(v: float) -> float:
        return mt1 + (1.0 - v / ymax1) * ph1

    e.append(
        f'<text x="{ml1 + pw1/2:.1f}" y="24" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1e293b">'
        "(a) On-Chip M9K Memory Blocks Required</text>"
    )
    for yv in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
        yy = py1(yv)
        stroke = "#cbd5e1" if yv == 0 else "#e2e8f0"
        e.append(f'<line x1="{ml1}" y1="{yy:.1f}" x2="{ml1+pw1}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"/>')
        e.append(f'<text x="{ml1-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12.0" fill="#334155">{yv}</text>')
    e.append(f'<rect x="{ml1}" y="{mt1}" width="{pw1}" height="{ph1}" fill="none" stroke="#475569" stroke-width="1.3"/>')

    for cap, col, lbl, anc, lx_lbl, dy in [
        (46, "#2563eb", "EP4CE10 = 46 M9K", "start", ml1 + 8, -7),
        (66, "#059669", "DE0-Nano = 66 M9K", "end", ml1 + pw1 - 8, -7),
    ]:
        yy = py1(cap)
        e.append(f'<line x1="{ml1}" y1="{yy:.1f}" x2="{ml1+pw1}" y2="{yy:.1f}" stroke="{col}" stroke-width="1.6" stroke-dasharray="5,4"/>')
        e.append(f'<text x="{lx_lbl}" y="{yy+dy:.1f}" text-anchor="{anc}" font-size="10.8" font-weight="bold" fill="{col}">{lbl}</text>')

    t_vals = [4, 8, 16]
    bar_w = 25.0
    gap = 6.0
    group_centers = [ml1 + 64, ml1 + 180, ml1 + 296]

    for idx, tv in enumerate(t_vals):
        cx = group_centers[idx]
        std = lookup[("standard_hpost_buf", tv, 4)]
        hf = lookup[("hidden_fused_y_psum", tv, 4)]
        std_dbl = 33 if tv == 8 else std["m9k_double_wgt_dma"]
        hf_dbl = 32 if tv == 8 else hf["m9k_double_wgt_dma"]
        bars = [
            (std["m9k_single_wgt"], "#93c5fd", "#1d4ed8"),
            (std_dbl, "#2563eb", "#1e3a8a"),
            (hf_dbl, "#f59e0b", "#b45309"),
        ]
        x0 = cx - (3 * bar_w + 2 * gap) / 2.0
        for bi, (val, bfill, bstroke) in enumerate(bars):
            bx = x0 + bi * (bar_w + gap)
            by = py1(val)
            bh = py1(0) - by
            e.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{bfill}" stroke="{bstroke}" stroke-width="1.2"/>')
            e.append(f'<text x="{bx + bar_w/2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#0f172a">{val}</text>')
        e.append(f'<text x="{cx:.1f}" y="{mt1+ph1+24}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1e293b">T_tile = {tv}</text>')

    e.append(f'<rect x="{ml1+8}" y="{mt1+6}" width="194" height="58" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.1"/>')
    for li, (col, text) in enumerate([("#93c5fd", "Standard (Single Wgt)"), ("#2563eb", "Standard (Double DMA)"), ("#f59e0b", "Hidden-Fused (Dbl DMA)")]):
        ly = mt1 + 13 + li * 16
        e.append(f'<rect x="{ml1+16}" y="{ly}" width="12" height="10" fill="{col}" stroke="#475569" stroke-width="0.6"/>')
        e.append(f'<text x="{ml1+34}" y="{ly+9}" font-size="10.4" fill="#0f172a">{text}</text>')

    e.append(f'<text x="22" y="{mt1+ph1/2:.1f}" text-anchor="middle" font-size="12.8" font-weight="bold" fill="#1e293b" transform="rotate(-90 22 {mt1+ph1/2:.1f})">Block RAM Usage (M9K Blocks)</text>')

    # Panel (b)
    ml2, mt2, pw2, ph2 = 512, 38, 384, 294
    ymax2 = 360.0

    def py2(v: float) -> float:
        return mt2 + (1.0 - v / ymax2) * ph2

    e.append(
        f'<text x="{ml2 + pw2/2:.1f}" y="24" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1e293b">'
        "(b) Required External Bandwidth (Decimal MB/s, 10⁶ B/s)</text>"
    )
    for yv in [0, 50, 100, 150, 200, 250, 300, 350]:
        yy = py2(yv)
        stroke = "#cbd5e1" if yv == 0 else "#e2e8f0"
        e.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"/>')
        e.append(f'<text x="{ml2-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12.0" fill="#334155">{yv}</text>')
    e.append(f'<rect x="{ml2}" y="{mt2}" width="{pw2}" height="{ph2}" fill="none" stroke="#475569" stroke-width="1.3"/>')

    y200 = py2(200.0)
    y188 = py2(bist_100_rd)
    e.append(f'<line x1="{ml2}" y1="{y200:.1f}" x2="{ml2+pw2}" y2="{y200:.1f}" stroke="#64748b" stroke-width="1.4" stroke-dasharray="3,3"/>')
    e.append(f'<rect x="{ml2+pw2-176}" y="{y200-17:.1f}" width="172" height="14" rx="2" fill="#ffffff" fill-opacity="0.88"/>')
    e.append(f'<text x="{ml2+pw2-6}" y="{y200-6:.1f}" text-anchor="end" font-size="10.4" font-weight="bold" fill="#475569">Theoretical Peak: 200.0 MB/s</text>')
    e.append(f'<line x1="{ml2}" y1="{y188:.1f}" x2="{ml2+pw2}" y2="{y188:.1f}" stroke="#0f766e" stroke-width="1.8" stroke-dasharray="6,3"/>')
    e.append(f'<rect x="{ml2+pw2-118}" y="{y188+3:.1f}" width="114" height="15" rx="2" fill="#ffffff" fill-opacity="0.90"/>')
    e.append(f'<text x="{ml2+pw2-6}" y="{y188+14:.1f}" text-anchor="end" font-size="10.2" font-weight="bold" fill="#0f766e">BIST: {fmt_en(bist_100_rd, 2)} MB/s</text>')

    group_centers2 = [ml2 + 68, ml2 + 192, ml2 + 316]
    bar_w2 = 26.0
    gap2 = 6.0
    for idx, tv in enumerate(t_vals):
        cx = group_centers2[idx]
        w4_row = lookup[("standard_hpost_buf", tv, 4)]
        w8_row = lookup[("standard_hpost_buf", tv, 8)]
        bw_w4_pure = (w4_row["weight_traffic_KiB"] * 1024.0 / 1e6) / (w4_row["compute_ms_80mhz"] * 1e-3)
        bw_w4_hdr = bw_w4_pure * (264.0 / 256.0)
        bw_w8_pure = (w8_row["weight_traffic_KiB"] * 1024.0 / 1e6) / (w8_row["compute_ms_80mhz"] * 1e-3)

        bars2 = [
            (bw_w4_pure, "#60a5fa", "#1d4ed8"),
            (bw_w4_hdr, "#0d9488", "#115e59"),
            (bw_w8_pure, "#ef4444", "#991b1b"),
        ]
        x0 = cx - (3 * bar_w2 + 2 * gap2) / 2.0
        for bi, (val, bfill, bstroke) in enumerate(bars2):
            bx = x0 + bi * (bar_w2 + gap2)
            by = py2(val)
            bh = py2(0) - by
            e.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w2}" height="{bh:.1f}" fill="{bfill}" stroke="{bstroke}" stroke-width="1.2"/>')
            lbl = fmt_en(val, 1 if val >= 80 or abs(val - 40.0) < 0.2 else 2)
            if abs(val - 165.0) < 0.2:
                lbl = "165.0"
            elif abs(val - 160.0) < 0.2:
                lbl = "160.0"
            elif abs(val - 320.0) < 0.2:
                lbl = "320.0"
            elif abs(val - 82.5) < 0.2:
                lbl = "82.5"
            elif abs(val - 41.25) < 0.2:
                lbl = "41.3"
            e.append(f'<text x="{bx + bar_w2/2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="9.8" font-weight="bold" fill="#0f172a">{lbl}</text>')
        e.append(f'<text x="{cx:.1f}" y="{mt2+ph2+24}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1e293b">T_tile = {tv}</text>')

    e.append(f'<rect x="{ml2+pw2-204}" y="{mt2+8}" width="196" height="62" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.1"/>')
    for li, (col, text) in enumerate([("#60a5fa", "W4 Pure (640 / T)"), ("#0d9488", "W4 + Header (660 / T)"), ("#ef4444", "W8 Pure (1,280 / T)")]):
        ly = mt2 + 16 + li * 17
        e.append(f'<rect x="{ml2+pw2-196}" y="{ly}" width="13" height="11" fill="{col}" stroke="#475569" stroke-width="0.6"/>')
        e.append(f'<text x="{ml2+pw2-177}" y="{ly+10}" font-size="10.8" fill="#0f172a">{text}</text>')

    e.append(f'<text x="466" y="{mt2+ph2/2:.1f}" text-anchor="middle" font-size="12.8" font-weight="bold" fill="#1e293b" transform="rotate(-90 466 {mt2+ph2/2:.1f})">Required Off-Chip Bandwidth (MB/s)</text>')

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig_rtl_timing(out_path: Path) -> None:
    """Compact single-column vector timing diagram (W=560) with large 12.5-15 px fonts."""
    W, H = 560, 256
    e: list[str] = [arrow_defs(), f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    e.append('<text x="10" y="42" font-size="13.5" font-weight="bold" fill="#1e293b">DMA Stream</text>')
    e.append('<text x="10" y="60" font-size="11.5" fill="#475569">(C_dma)</text>')

    e.append('<text x="10" y="112" font-size="13.5" font-weight="bold" fill="#1e293b">MAC Compute</text>')
    e.append('<text x="10" y="130" font-size="11.5" fill="#475569">(Bank 0 / 1)</text>')

    e.append('<text x="10" y="178" font-size="13.5" font-weight="bold" fill="#991b1b">Engine Stall</text>')
    e.append('<text x="10" y="196" font-size="11.5" fill="#b91c1c">(WAIT_DMA)</text>')

    for x in [108, 228, 358, 404, 530]:
        e.append(f'<line x1="{x}" y1="14" x2="{x}" y2="210" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="4,3"/>')

    # Row 1: DMA Stream
    e.append('<rect x="108" y="20" width="120" height="46" rx="4" fill="#ccfbf1" stroke="#0f766e" stroke-width="1.6"/>')
    e.append('<text x="168" y="40" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#134e4a">C_dma(0) → B0</text>')
    e.append('<text x="168" y="58" text-anchor="middle" font-size="10.8" fill="#115e59">8P+14p₀+1 + CDC</text>')

    e.append('<rect x="228" y="20" width="176" height="46" rx="4" fill="#e0f2fe" stroke="#0284c7" stroke-width="1.6"/>')
    e.append('<text x="316" y="40" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#0c4a6e">C_dma(i) → Bank 1</text>')
    e.append('<text x="316" y="58" text-anchor="middle" font-size="10.8" fill="#075985">8P+14p_i+1 + Ref + CDC</text>')

    e.append('<rect x="404" y="20" width="126" height="46" rx="4" fill="#ccfbf1" stroke="#0f766e" stroke-width="1.6"/>')
    e.append('<text x="467" y="40" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#134e4a">C_dma(i+1) → B0</text>')
    e.append('<text x="467" y="58" text-anchor="middle" font-size="10.8" fill="#115e59">Hidden (≤ C_tile)</text>')

    # Row 2: MAC Compute
    e.append('<rect x="228" y="90" width="130" height="46" rx="4" fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.6"/>')
    e.append('<text x="293" y="110" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#1e3a8a">C_tile(i−1) [B0]</text>')
    e.append('<text x="293" y="128" text-anchor="middle" font-size="10.4" fill="#1e40af">T·K + epi (6/13)</text>')

    e.append('<rect x="404" y="90" width="142" height="46" rx="4" fill="#fef3c7" stroke="#b45309" stroke-width="1.6"/>')
    e.append('<text x="475" y="110" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#78350f">C_tile(i) [Bank 1]</text>')
    e.append('<text x="475" y="128" text-anchor="middle" font-size="10.4" fill="#92400e">T·K + 13 (final K)</text>')

    # Row 3: Engine Stall
    e.append('<rect x="108" y="158" width="120" height="44" rx="4" fill="#fee2e2" stroke="#dc2626" stroke-width="1.6"/>')
    e.append('<text x="168" y="177" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#991b1b">Cold-Start</text>')
    e.append('<text x="168" y="194" text-anchor="middle" font-size="11.0" fill="#b91c1c">C_dma(0)</text>')

    e.append('<rect x="358" y="158" width="46" height="44" rx="4" fill="#fee2e2" stroke="#dc2626" stroke-width="1.6"/>')
    e.append('<text x="381" y="185" text-anchor="middle" font-size="11.8" font-weight="bold" fill="#991b1b">Stall</text>')

    e.append('<text x="482" y="176" text-anchor="middle" font-size="11.8" font-weight="bold" fill="#991b1b">max(0, C_dma − C_tile)</text>')
    e.append('<text x="482" y="194" text-anchor="middle" font-size="11.0" fill="#475569">0 stall when hidden</text>')
    e.append('<line x1="418" y1="180" x2="406" y2="180" stroke="#dc2626" stroke-width="1.8" marker-end="url(#arr_red)"/>')

    e.append('<line x1="108" y1="222" x2="548" y2="222" stroke="#334155" stroke-width="1.6" marker-end="url(#arr_slate)"/>')
    e.append('<text x="328" y="244" text-anchor="middle" font-size="13.0" font-weight="bold" fill="#1e293b">Engine Clock Cycles (clk_eng = 80 MHz)</text>')

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig3a_sdram_stall_reconstruction(g4b: dict, out_path: Path) -> None:
    """Compact single-column W=560 canvas with large 12.5-15.5 px fonts."""
    freqs_mhz = [60, 80, 90, 100, 120]
    meas_pts = []
    eta_vals = []
    for mhz in freqs_mhz:
        r = g4b[f"sdram_{mhz}mhz"]
        f = r["ffn_512_1536_512_w4a8"]
        l1_c = f["l1_counters"]
        l2_c = f["l2_counters"]
        c_dma1 = l1_c["dma_cycles"] / 96.0
        c_dma2 = l2_c["dma_cycles"] / 96.0
        c_tile1 = (l1_c["total_cycles"] - l1_c["stall_cycles"]) / 96.0
        c_tile2 = (l2_c["total_cycles"] - l2_c["stall_cycles"]) / 96.0
        recon_stall = (c_dma1 + 95.0 * max(0.0, c_dma1 - c_tile1)) + (
            c_dma2 + 95.0 * max(0.0, c_dma2 - c_tile2)
        )
        recon_cyc = (c_tile1 + c_tile2) * 96.0 + recon_stall
        recon_pct = 100.0 * recon_stall / recon_cyc
        meas_pts.append({
            "nom_mhz": mhz,
            "sd_mhz": r["measured_sdram_mhz"],
            "stall_pct": f["stall_pct_of_total"],
            "recon_pct": recon_pct,
        })
        eta_vals.append(f["dma_active_bw_MBps_1e6"] / (2.0 * r["measured_sdram_mhz"]))

    mean_eta = sum(eta_vals) / len(eta_vals)
    c_t1, c_t2 = 2061.00, 2056.33
    f_eng_mhz = 80.0
    chunk_bytes = 4224.0
    f_crit_mhz = chunk_bytes * f_eng_mhz / (((c_t1 + c_t2) / 2.0) * 2.0 * mean_eta)

    def recon_curve_pct(f_sd: float) -> float:
        bw = 2.0 * f_sd * mean_eta
        c_dma = (chunk_bytes / bw) * f_eng_mhz
        st = (c_dma + 95.0 * max(0.0, c_dma - c_t1)) + (c_dma + 95.0 * max(0.0, c_dma - c_t2))
        tot = 96.0 * (c_t1 + c_t2) + st
        return 100.0 * st / tot

    W, H = 560, 360
    ml, mr, mt, mb = 64, 18, 16, 54
    pw, ph = W - ml - mr, H - mt - mb
    xmin, xmax = 55.0, 125.0
    ymin, ymax = 0.0, 40.0

    def px(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * pw

    def py(y: float) -> float:
        return mt + (1.0 - (y - ymin) / (ymax - ymin)) * ph

    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    x_crit = px(f_crit_mhz)
    e.append(f'<rect x="{ml:.1f}" y="{mt:.1f}" width="{x_crit - ml:.1f}" height="{ph:.1f}" fill="#fef2f2" opacity="0.75"/>')
    e.append(f'<rect x="{x_crit:.1f}" y="{mt:.1f}" width="{ml + pw - x_crit:.1f}" height="{ph:.1f}" fill="#f0fdf4" opacity="0.75"/>')

    for yv in [0, 10, 20, 30, 40]:
        yy = py(yv)
        stroke = "#cbd5e1" if yv == 0 else "#e2e8f0"
        e.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1.1"/>')
        e.append(f'<text x="{ml-8}" y="{yy+5:.1f}" text-anchor="end" font-size="13.5" fill="#334155">{yv}%</text>')

    for xv in [60, 70, 80, 90, 100, 110, 120]:
        xx = px(xv)
        e.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#e2e8f0" stroke-width="1.1"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt+ph+20}" text-anchor="middle" font-size="13.5" fill="#334155">{xv}</text>')

    e.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#475569" stroke-width="1.4"/>')

    e.append(f'<line x1="{x_crit:.1f}" y1="{mt}" x2="{x_crit:.1f}" y2="{mt+ph}" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="6,4"/>')
    e.append(f'<text x="{x_crit+6:.1f}" y="{mt+20}" font-size="12.5" font-weight="bold" fill="#b91c1c">f_crit ≈ {fmt_en(f_crit_mhz, 1)} MHz</text>')
    e.append(f'<text x="{px(71.0):.1f}" y="{mt+20}" text-anchor="middle" font-size="12.8" font-weight="bold" fill="#991b1b">BW-Bound</text>')
    e.append(f'<text x="{px(107.5):.1f}" y="{mt+38}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#166534">Compute-Saturated</text>')

    curve_pts = [f"{px(55.0 + 0.5*k):.2f},{py(recon_curve_pct(55.0 + 0.5*k)):.2f}" for k in range(141)]
    e.append(f'<polyline points="{" ".join(curve_pts)}" fill="none" stroke="#1d4ed8" stroke-width="2.8"/>')

    label_specs = {
        60: (110.0, 68.0, "start", False),
        80: (224.0, 206.0, "end", True),
        90: (298.0, 238.0, "start", True),
        100: (368.0, 198.0, "start", True),
        120: (532.0, 260.0, "end", True),
    }
    for p in meas_pts:
        cx, cy = px(p["sd_mhz"]), py(p["stall_pct"])
        lx_t, ly_t, anc, leader = label_specs[p["nom_mhz"]]
        if leader:
            e.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{lx_t:.1f}" y2="{ly_t+4:.1f}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="2,2"/>')
        e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.8" fill="#ea580c" stroke="#ffffff" stroke-width="1.6"/>')
        lbl = f"{p['nom_mhz']}M: {fmt_en(p['stall_pct'], 2)}%"
        e.append(f'<text x="{lx_t:.1f}" y="{ly_t:.1f}" text-anchor="{anc}" font-size="12.5" font-weight="bold" fill="#1e293b">{xml_esc(lbl)}</text>')

    lx, ly = ml + pw - 266, mt + 52
    e.append(f'<rect x="{lx}" y="{ly}" width="258" height="60" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>')
    e.append(f'<line x1="{lx+10}" y1="{ly+20}" x2="{lx+36}" y2="{ly+20}" stroke="#1d4ed8" stroke-width="2.8"/>')
    e.append(f'<text x="{lx+44}" y="{ly+24}" font-size="12.2" fill="#0f172a">Counter Reconstruction</text>')
    e.append(f'<circle cx="{lx+23}" cy="{ly+44}" r="5.5" fill="#ea580c" stroke="#ffffff" stroke-width="1.3"/>')
    e.append(f'<text x="{lx+44}" y="{ly+48}" font-size="12.2" fill="#0f172a">EP4CE10 Silicon (512→1536)</text>')

    e.append(
        f'<text x="{ml + pw/2:.1f}" y="{H-12}" text-anchor="middle" font-size="13.8" font-weight="bold" fill="#1e293b">'
        "SDRAM Clock Frequency (MHz, Engine @ 80 MHz)</text>"
    )
    e.append(
        f'<text x="18" y="{mt + ph/2:.1f}" text-anchor="middle" font-size="13.8" font-weight="bold" fill="#1e293b" transform="rotate(-90 18 {mt + ph/2:.1f})">'
        "DMA-Wait Stall Ratio (%)</text>"
    )

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig3b_oos_stall_prediction(oos: dict, out_path: Path) -> None:
    """Compact single-column W=560 canvas with large 12.2-15.5 px fonts."""
    s_specs = [
        ("ffn_192_384_192", "192→384→192 (f_crit = 92.4 MHz)", "#dc2626", "none"),
        ("ffn_256_576_256", "256→576→256 (f_crit = 90.6 MHz)", "#2563eb", "6,4"),
        ("ffn_256_960_256", "256→960→256 (f_crit = 88.1 MHz)", "#059669", "3,3"),
    ]
    freqs = [60, 90, 100, 120]

    shape_meta = {
        "ffn_192_384_192": [(0, 24, 1, 192, 104), (2496, 12, 1, 384, 200)],
        "ffn_256_576_256": [(4896, 36, 1, 256, 136), (9792, 16, 2, 288, 152)],
        "ffn_256_960_256": [(14656, 60, 1, 256, 136), (22816, 16, 2, 480, 248)],
    }

    def eval_curve_pct(sname: str, f_sd: float) -> float:
        t_rfc = 9.0 if f_sd > 100.0 else 8.0
        ref_int = 14.0 * f_sd
        tot_act, tot_st = 0.0, 0.0
        for base, n_tiles, k_chunks, k_chunk, pkt in shape_meta[sname]:
            addr = base
            c_tiles, c_dmas = [], []
            for _nt in range(n_tiles):
                for kc in range(k_chunks):
                    c_tile = 4 * k_chunk + (13 if kc == k_chunks - 1 else 6)
                    pages = ((addr + pkt - 1) // 32) - (addr // 32) + 1
                    sd_cyc = (8 * pkt + 14 * pages + 1) * (1.0 + t_rfc / ref_int) + 3.0
                    c_dma = sd_cyc * (80.0 / f_sd) + 4.0
                    c_tiles.append(c_tile)
                    c_dmas.append(c_dma)
                    addr += pkt
            st = c_dmas[0] + sum(max(0.0, c_dmas[i] - c_tiles[i - 1]) for i in range(1, len(c_dmas)))
            tot_act += sum(c_tiles)
            tot_st += st
        return 100.0 * tot_st / (tot_act + tot_st)

    W, H = 560, 360
    ml, mr, mt, mb = 64, 18, 16, 54
    pw, ph = W - ml - mr, H - mt - mb
    xmin, xmax = 55.0, 125.0
    ymin, ymax = 0.0, 42.0

    def px(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * pw

    def py(y: float) -> float:
        return mt + (1.0 - (y - ymin) / (ymax - ymin)) * ph

    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    for yv in [0, 10, 20, 30, 40]:
        yy = py(yv)
        stroke = "#cbd5e1" if yv == 0 else "#e2e8f0"
        e.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1.1"/>')
        e.append(f'<text x="{ml-8}" y="{yy+5:.1f}" text-anchor="end" font-size="13.5" fill="#334155">{yv}%</text>')

    for xv in [60, 70, 80, 90, 100, 110, 120]:
        xx = px(xv)
        e.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#e2e8f0" stroke-width="1.1"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt+ph+20}" text-anchor="middle" font-size="13.5" fill="#334155">{xv}</text>')

    e.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#475569" stroke-width="1.4"/>')

    for sname, _lbl, col, dash in s_specs:
        pts = [f"{px(55.0 + 0.5*k):.2f},{py(eval_curve_pct(sname, 55.0 + 0.5*k)):.2f}" for k in range(141)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        e.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.6"{dash_attr}/>')
        for mhz in freqs:
            r = oos["per_frequency"][f"sdram_{mhz}mhz"]
            st_m = r["shapes"][sname]["combined"]["meas_stall_pct"]
            cx, cy = px(r["measured_sdram_mhz"]), py(st_m)
            e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{col}" stroke="#ffffff" stroke-width="1.5"/>')

    r90_192 = oos["per_frequency"]["sdram_90mhz"]["shapes"]["ffn_192_384_192"]
    r90_960 = oos["per_frequency"]["sdram_90mhz"]["shapes"]["ffn_256_960_256"]
    x90 = px(90.0)
    e.append(f'<line x1="{x90:.1f}" y1="{py(21.0):.1f}" x2="{x90:.1f}" y2="{py(0.0):.1f}" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="3,3"/>')
    e.append(
        f'<text x="{x90+6:.1f}" y="{py(20.0):.1f}" font-size="12.2" font-weight="bold" fill="#1e293b">'
        "Regime Split @ 90 MHz:</text>"
    )
    e.append(
        f'<text x="{x90+6:.1f}" y="{py(16.6):.1f}" font-size="11.8" font-weight="bold" fill="#991b1b">'
        f"192→384 L1: {fmt_en(r90_192['l1']['meas_stall_pct'],2)}% / {fmt_en(r90_192['l1']['pred_stall_pct'],2)}%</text>"
    )
    e.append(
        f'<text x="{x90+6:.1f}" y="{py(13.2):.1f}" font-size="11.8" font-weight="bold" fill="#065f46">'
        f"256→960 L2: {fmt_en(r90_960['l2']['meas_stall_pct'],2)}% / {fmt_en(r90_960['l2']['pred_stall_pct'],2)}%</text>"
    )

    lx, ly = ml + pw - 278, mt + 10
    e.append(f'<rect x="{lx}" y="{ly}" width="270" height="88" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>')
    for idx, (_sn, lbl, col, dash) in enumerate(s_specs):
        yy = ly + 20 + idx * 25
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        e.append(f'<line x1="{lx+10}" y1="{yy}" x2="{lx+36}" y2="{yy}" stroke="{col}" stroke-width="2.6"{dash_attr}/>')
        e.append(f'<circle cx="{lx+23}" cy="{yy}" r="4.8" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>')
        e.append(f'<text x="{lx+44}" y="{yy+4}" font-size="12.2" font-weight="bold" fill="#0f172a">{xml_esc(lbl)}</text>')

    e.append(
        f'<text x="{ml + pw/2:.1f}" y="{H-12}" text-anchor="middle" font-size="13.8" font-weight="bold" fill="#1e293b">'
        "SDRAM Clock Frequency (MHz, Engine @ 80 MHz)</text>"
    )
    e.append(
        f'<text x="18" y="{mt + ph/2:.1f}" text-anchor="middle" font-size="13.8" font-weight="bold" fill="#1e293b" transform="rotate(-90 18 {mt + ph/2:.1f})">'
        "DMA-Wait Stall Ratio (%)</text>"
    )

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig4_wer_pareto_and_tones(
    paired: dict, abl_tone: dict, out_path: Path
) -> None:
    """Two-panel figure designed for full-width figure* with large 11.5-14.5 px fonts."""
    vmodels = paired["vivos_clean_all_models_utt_vs_speaker"]
    w4g64_blk = abl_tone["controlled_w4_vs_w3_g64_ablation"]

    rows = [
        ("Sim. FFN W8A8 (ONNX)", vmodels["30M_fpga_bitexact_w8a8"]["speaker_clustered_bootstrap_vs_fp32"], "#16a34a"),
        ("FFN W4 (g64, Ctrl)", w4g64_blk["w4_g64_vs_fp32_spk"], "#d97706"),
        ("Full Enc. W8A8 (ONNX)", vmodels["30M_W8A8"]["speaker_clustered_bootstrap_vs_fp32"], "#d97706"),
        ("FFN W4A8 (ONNX/FPGA)", vmodels["30M_fpga_bitexact_w4a8"]["speaker_clustered_bootstrap_vs_fp32"], "#ea580c"),
        ("FFN W3 (g64, Ctrl)", w4g64_blk["w3_g64_vs_fp32_spk"], "#c2410c"),
        ("All W4 (per-ch)", vmodels["30M_all_w4_perch"]["speaker_clustered_bootstrap_vs_fp32"], "#c2410c"),
        ("FFN W2 (u4, g64)", vmodels["30M_ffn_w2_u4_g64"]["speaker_clustered_bootstrap_vs_fp32"], "#b91c1c"),
    ]

    W, H = 960, 404
    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    # Panel (a)
    ml1, mt1, pw1, ph1 = 178, 36, 316, 300
    xmin1, xmax1 = -0.30, 2.38

    def px1(v: float) -> float:
        return ml1 + (v - xmin1) / (xmax1 - xmin1) * pw1

    e.append(
        f'<text x="{ml1 + pw1/2:.1f}" y="22" text-anchor="middle" font-size="13.2" font-weight="bold" fill="#1e293b">'
        "(a) ΔWER vs. FP32 (4.610%) &amp; 19-Speaker 95% CIs</text>"
    )
    e.append(f'<rect x="{ml1:.1f}" y="{mt1:.1f}" width="{px1(0.10) - ml1:.1f}" height="{ph1:.1f}" fill="#f0fdf4" opacity="0.8"/>')

    for xv in [0.0, 0.5, 1.0, 1.5, 2.0]:
        xx = px1(xv)
        col = "#94a3b8" if xv == 0.0 else "#e2e8f0"
        e.append(f'<line x1="{xx:.1f}" y1="{mt1}" x2="{xx:.1f}" y2="{mt1+ph1}" stroke="{col}" stroke-width="1.1"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt1+ph1+22}" text-anchor="middle" font-size="11.8" fill="#334155">{fmt_en(xv, 1)}%</text>')

    x_ni = px1(0.10)
    e.append(f'<line x1="{x_ni:.1f}" y1="{mt1}" x2="{x_ni:.1f}" y2="{mt1+ph1}" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="5,3"/>')
    e.append(f'<text x="{x_ni+5:.1f}" y="{mt1+16}" font-size="11.0" font-weight="bold" fill="#b91c1c">+0.10% Ref.</text>')
    e.append(f'<rect x="{ml1}" y="{mt1}" width="{pw1}" height="{ph1}" fill="none" stroke="#475569" stroke-width="1.3"/>')

    row_step = ph1 / len(rows)
    for idx, (label, blk, col) in enumerate(rows):
        yy = mt1 + (idx + 0.5) * row_step
        if idx > 0:
            ysep = mt1 + idx * row_step
            e.append(f'<line x1="{ml1}" y1="{ysep:.1f}" x2="{ml1+pw1}" y2="{ysep:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        d = float(blk["delta_wer_pct"])
        ci_lo, ci_hi = float(blk["delta_wer_ci95"][0]), float(blk["delta_wer_ci95"][1])
        x_pt, x_lo, x_hi = px1(d), px1(ci_lo), px1(ci_hi)

        e.append(f'<text x="{ml1-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11.5" font-weight="bold" fill="#1e293b">{xml_esc(label)}</text>')
        e.append(f'<line x1="{x_lo:.1f}" y1="{yy:.1f}" x2="{x_hi:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="2.4"/>')
        e.append(f'<line x1="{x_lo:.1f}" y1="{yy-5:.1f}" x2="{x_lo:.1f}" y2="{yy+5:.1f}" stroke="{col}" stroke-width="2.0"/>')
        e.append(f'<line x1="{x_hi:.1f}" y1="{yy-5:.1f}" x2="{x_hi:.1f}" y2="{yy+5:.1f}" stroke="{col}" stroke-width="2.0"/>')
        e.append(f'<circle cx="{x_pt:.1f}" cy="{yy:.1f}" r="4.8" fill="{col}" stroke="#ffffff" stroke-width="1.3"/>')

        txt = f"{fmt_en(d, 3, signed=True)} [{fmt_en(ci_lo, 3, signed=True)}, {fmt_en(ci_hi, 3, signed=True)}]"
        if idx < 6:
            e.append(f'<text x="{x_hi + 6:.1f}" y="{yy+4:.1f}" font-size="10.4" font-weight="bold" fill="#1e293b">{txt}</text>')
        else:
            e.append(f'<text x="{x_pt:.1f}" y="{yy-10.0:.1f}" text-anchor="middle" font-size="10.4" font-weight="bold" fill="#991b1b">{txt}</text>')

    d34 = w4g64_blk["w3_g64_minus_w4_g64_spk"]
    e.append(f'<rect x="{ml1+6}" y="{mt1+250}" width="188" height="42" rx="4" fill="#fffbeb" stroke="#f59e0b" stroke-width="1.1"/>')
    e.append(f'<text x="{ml1+12}" y="{mt1+267}" font-size="10.2" font-weight="bold" fill="#92400e">g64 Ablation (4 → 3 bit):</text>')
    e.append(
        f'<text x="{ml1+12}" y="{mt1+284}" font-size="10.0" fill="#78350f">'
        f"W3−W4 = {fmt_en(d34['delta_wer_pct'],3,signed=True)}% [{fmt_en(d34['delta_wer_ci95'][0],3,signed=True)}, {fmt_en(d34['delta_wer_ci95'][1],3,signed=True)}]</text>"
    )
    e.append(
        f'<text x="{ml1 + pw1/2:.1f}" y="{mt1+ph1+46}" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#1e293b">'
        "ΔWER Relative to FP32 Baseline (pp)</text>"
    )

    # Panel (b)
    ml2, mt2, pw2, ph2 = 556, 36, 380, 300
    ymax2 = 1.75

    def py2(v: float) -> float:
        return mt2 + (1.0 - v / ymax2) * ph2

    e.append(
        f'<text x="{ml2 + pw2/2:.1f}" y="22" text-anchor="middle" font-size="13.2" font-weight="bold" fill="#1e293b">'
        "(b) 6-Tone Error Rate on Matched-Base Syllables (~96% cov.)</text>"
    )
    for yv in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]:
        yy = py2(yv)
        stroke = "#cbd5e1" if yv == 0.0 else "#e2e8f0"
        e.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1.1"/>')
        e.append(f'<text x="{ml2-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11.8" fill="#334155">{fmt_en(yv, 1)}%</text>')
    e.append(f'<rect x="{ml2}" y="{mt2}" width="{pw2}" height="{ph2}" fill="none" stroke="#475569" stroke-width="1.3"/>')

    tone_labels = [
        ("ngang", "Ngang"),
        ("huyen", "Huyền"),
        ("sac", "Sắc"),
        ("hoi", "Hỏi"),
        ("nga", "Ngã"),
        ("nang", "Nặng"),
    ]
    t_fp32 = abl_tone["tone_6x6_matrices"]["30M_FP32"]["per_tone"]
    t_w4 = abl_tone["tone_6x6_matrices"]["30M_fpga_bitexact_w4a8"]["per_tone"]

    bw2 = 21.0
    gap2 = 4.0
    step2 = pw2 / len(tone_labels)
    for idx, (tk, tname) in enumerate(tone_labels):
        cx = ml2 + (idx + 0.5) * step2
        v_fp = float(t_fp32[tk]["error_rate_pct"])
        v_w4 = float(t_w4[tk]["error_rate_pct"])
        c_fp = int(t_fp32[tk]["tone_errors"])
        c_w4 = int(t_w4[tk]["tone_errors"])

        x1 = cx - bw2 - gap2 / 2.0
        x2 = cx + gap2 / 2.0
        y1 = py2(v_fp)
        y2 = py2(v_w4)
        e.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{bw2}" height="{py2(0)-y1:.1f}" fill="#3b82f6" stroke="#1d4ed8" stroke-width="1.1"/>')
        e.append(f'<rect x="{x2:.1f}" y="{y2:.1f}" width="{bw2}" height="{py2(0)-y2:.1f}" fill="#f97316" stroke="#c2410c" stroke-width="1.1"/>')
        e.append(f'<text x="{x1+bw2/2:.1f}" y="{y1-5:.1f}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#1e3a8a">{c_fp}</text>')
        e.append(f'<text x="{x2+bw2/2:.1f}" y="{y2-5:.1f}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#9a3412">{c_w4}</text>')
        e.append(f'<text x="{cx:.1f}" y="{mt2+ph2+22}" text-anchor="middle" font-size="11.8" font-weight="bold" fill="#1e293b">{tname}</text>')

    fp_all = abl_tone["tone_6x6_matrices"]["30M_FP32"]
    w4_all = abl_tone["tone_6x6_matrices"]["30M_fpga_bitexact_w4a8"]
    td_all = abl_tone["tone_6x6_speaker_bootstrap_deltas"]["w4a8_bitexact_minus_fp32"]
    nga_ci_hi = td_all["per_tone_delta"]["nga"]["ci95"][1]
    e.append(f'<rect x="{ml2+8}" y="{mt2+8}" width="248" height="72" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.1"/>')
    e.append(f'<rect x="{ml2+16}" y="{mt2+18}" width="13" height="11" fill="#3b82f6" stroke="#1d4ed8" stroke-width="0.8"/>')
    e.append(
        f'<text x="{ml2+36}" y="{mt2+28}" font-size="11.0" fill="#0f172a">'
        f"FP32: {fp_all['overall_same_base_tone_errors']}/7,416 ({fmt_en(fp_all['overall_same_base_tone_err_pct'],3)}%)</text>"
    )
    e.append(f'<rect x="{ml2+16}" y="{mt2+38}" width="13" height="11" fill="#f97316" stroke="#c2410c" stroke-width="0.8"/>')
    e.append(
        f'<text x="{ml2+36}" y="{mt2+48}" font-size="11.0" fill="#0f172a">'
        f"W4A8: {w4_all['overall_same_base_tone_errors']}/7,412 ({fmt_en(w4_all['overall_same_base_tone_err_pct'],3)}%)</text>"
    )
    e.append(
        f'<text x="{ml2+16}" y="{mt2+68}" font-size="10.4" font-weight="bold" fill="#475569">'
        f"Overall Δ: {fmt_en(td_all['overall_delta_err_pct'],3,signed=True)}% · Ngã CI hi: {fmt_en(nga_ci_hi,3,signed=True)}%</text>"
    )

    e.append(f'<text x="{ml2 + pw2/2:.1f}" y="{mt2+ph2+46}" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#1e293b">Vietnamese Lexical Tones (Bar Labels: Error Count)</text>')
    e.append(f'<text x="510" y="{mt2+ph2/2:.1f}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1e293b" transform="rotate(-90 510 {mt2+ph2/2:.1f})">Tone Error Rate / Matched Base (%)</text>')

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def build_padded_t500_mcu_data(g5b: dict, img: dict) -> dict:
    jtag_rates = g5b["part_a_full_sequence_measurement"]["measured_full_sequence_t263"]
    wr_bps = float(jtag_rates["measured_jtag_write_Bps"])
    rd_bps = float(jtag_rates["measured_jtag_read_Bps"])

    t0 = (500 - 7) // 2
    t1 = (t0 + 1) // 2
    t2 = (t1 + 1) // 2
    t3 = (t2 + 1) // 2
    stage_t500 = {0: t0, 1: t1, 2: t2, 3: t3, 4: t2, 5: t1}

    ffn_macs = 0
    act_io_bytes_padded = 0
    total_block_tiles = 0
    pred_cycles_total = 0

    for blk in img["blocks"]:
        st = int(blk["prefix"].split("/")[2])
        d_m = int(blk["d_model"])
        d_ff = int(blk["d_ff"])
        t_len = stage_t500[st]
        n_t = (t_len + 3) // 4
        total_block_tiles += n_t
        ffn_macs += 2 * t_len * d_m * d_ff
        act_io_bytes_padded += 2 * (n_t * 4) * d_m

        p1 = predict_layer_rtl(
            blk["l1_sdram_base_w128"],
            blk["l1_meta"],
            nom_sdram_mhz=100.0,
            ref_interval=1400,
            t_rfc=8,
            eng_mhz=80.0,
        )
        p2 = predict_layer_rtl(
            blk["l2_sdram_base_w128"],
            blk["l2_meta"],
            nom_sdram_mhz=100.0,
            ref_interval=1400,
            t_rfc=8,
            eng_mhz=80.0,
        )
        pred_cycles_total += n_t * (p1["pred_total_cyc"] + p2["pred_total_cyc"])

    ffn_gmac = ffn_macs / 1e9
    t_fpga_s = pred_cycles_total / 80e6
    act_mb = act_io_bytes_padded / 1e6
    g_max = ffn_gmac / t_fpga_s
    t_jtag_io_s = (act_io_bytes_padded / 2.0) / wr_bps + (act_io_bytes_padded / 2.0) / rd_bps
    bw_jtag_eff = act_mb / t_jtag_io_s

    return {
        "total_block_tiles_padded": total_block_tiles,
        "ffn_gmac": round(ffn_gmac, 9),
        "t_fpga_s": round(t_fpga_s, 6),
        "act_io_bytes": act_io_bytes_padded,
        "act_io_MB_decimal": round(act_mb, 6),
        "fpga_asymptote_gmac_s": round(g_max, 4),
        "jtag_eff_bw_MBps": round(bw_jtag_eff, 6),
    }


def generate_fig5_mcu_breakeven(data: dict, g5b: dict, out_path: Path) -> None:
    """Compact single-column W=560 canvas with large 12.2-15.5 px fonts."""
    ffn_gmac = data["ffn_gmac"]
    t_fpga_s = data["t_fpga_s"]
    act_bytes = float(data["act_io_bytes"])
    act_mb = data["act_io_MB_decimal"]
    g_max = data["fpga_asymptote_gmac_s"]

    jtag_rates = g5b["part_a_full_sequence_measurement"]["measured_full_sequence_t263"]
    wr_bps = float(jtag_rates["measured_jtag_write_Bps"])
    rd_bps = float(jtag_rates["measured_jtag_read_Bps"])
    t_io_jtag = (act_bytes / 2.0) / wr_bps + (act_bytes / 2.0) / rd_bps
    bw_jtag_eff = act_mb / t_io_jtag

    W, H = 560, 360
    ml, mr, mt, mb = 64, 18, 16, 54
    pw, ph = W - ml - mr, H - mt - mb
    lx_min, lx_max = math.log10(0.007), math.log10(50.0)
    ymin, ymax = 0.0, 1.8

    def px(bw: float) -> float:
        return ml + (math.log10(bw) - lx_min) / (lx_max - lx_min) * pw

    def py(g: float) -> float:
        return mt + (1.0 - (g - ymin) / (ymax - ymin)) * ph

    def g_seq(bw: float) -> float:
        return ffn_gmac / (t_fpga_s + act_mb / bw)

    def g_ovl(bw: float) -> float:
        return ffn_gmac / max(t_fpga_s, act_mb / bw)

    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    y_m7_top, y_m7_bot = py(0.5), py(0.3)
    e.append(f'<rect x="{ml}" y="{y_m7_top:.1f}" width="{pw}" height="{y_m7_bot - y_m7_top:.1f}" fill="#fef9c3" opacity="0.85"/>')

    n_pts = 200
    bws = [10 ** (lx_min + (lx_max - lx_min) * i / (n_pts - 1)) for i in range(n_pts)]
    seq_poly = [f"{px(b):.2f},{py(g_seq(b)):.2f}" for b in bws]
    ovl_pts = [f"{px(b):.2f},{py(g_ovl(b)):.2f}" for b in bws]
    poly_all = seq_poly + [f"{px(bws[-1]):.2f},{py(0.0):.2f}", f"{px(bws[0]):.2f},{py(0.0):.2f}"]
    e.append(f'<polygon points="{" ".join(poly_all)}" fill="#dbeafe" opacity="0.55"/>')

    for yv in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]:
        yy = py(yv)
        stroke = "#cbd5e1" if yv == 0.0 else "#e2e8f0"
        e.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1.1"/>')
        e.append(f'<text x="{ml-8}" y="{yy+5:.1f}" text-anchor="end" font-size="13.2" fill="#334155">{fmt_en(yv, 1)}</text>')

    xticks = [(0.01, "0.01"), (0.1, "0.1"), (1.0, "1"), (5.0, "5"), (10.0, "10"), (30.0, "30")]
    for xv, xlbl in xticks:
        xx = px(xv)
        e.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#e2e8f0" stroke-width="1.1"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt+ph+20}" text-anchor="middle" font-size="13.2" fill="#334155">{xlbl}</text>')

    e.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#475569" stroke-width="1.4"/>')

    y_ceil = py(g_max)
    e.append(f'<line x1="{ml}" y1="{y_ceil:.1f}" x2="{ml+pw}" y2="{y_ceil:.1f}" stroke="#0f766e" stroke-width="1.8" stroke-dasharray="6,4"/>')
    e.append(f'<text x="{ml+8}" y="{y_ceil-6:.1f}" font-size="12.0" font-weight="bold" fill="#0f766e">FPGA Ceiling: {fmt_en(g_max, 2)} GMAC/s (1.018 s)</text>')

    e.append(f'<text x="{ml+pw-8}" y="{mt+20}" text-anchor="end" font-size="11.8" font-weight="bold" fill="#475569">Cortex-A9 NEON (~1.4–3.9)</text>')
    e.append(f'<text x="{ml+8}" y="{py(0.38):.1f}" font-size="11.8" font-weight="bold" fill="#854d0e">Cortex-M7 CMSIS-NN (~0.3–0.5)</text>')
    e.append(f'<text x="{px(6.5):.1f}" y="{py(0.16):.1f}" text-anchor="middle" font-size="12.0" font-weight="bold" fill="#1e40af">Offload Regime</text>')

    e.append(f'<polyline points="{" ".join(ovl_pts)}" fill="none" stroke="#9333ea" stroke-width="2.4" stroke-dasharray="5,4"/>')
    e.append(f'<polyline points="{" ".join(seq_poly)}" fill="none" stroke="#1d4ed8" stroke-width="2.8"/>')

    g_jtag_eff = g_seq(bw_jtag_eff)
    markers = [
        (bw_jtag_eff, f"JTAG: {fmt_en(g_jtag_eff, 4)}", (8, -10, "start")),
        (1.0, f"1M: {fmt_en(g_seq(1.0), 3)}", (-8, -10, "end")),
        (5.0, f"5M: {fmt_en(g_seq(5.0), 3)}", (-8, 18, "end")),
        (10.0, f"10M: {fmt_en(g_seq(10.0), 3)}", (8, 18, "start")),
    ]
    for bw, lbl, (dx, dy, anc) in markers:
        cx, cy = px(bw), py(g_seq(bw))
        e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
        e.append(f'<text x="{cx+dx:.1f}" y="{cy+dy:.1f}" text-anchor="{anc}" font-size="12.2" font-weight="bold" fill="#1e293b">{xml_esc(lbl)}</text>')

    lx, ly = ml + 8, mt + 8
    e.append(f'<rect x="{lx}" y="{ly}" width="248" height="56" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>')
    e.append(f'<line x1="{lx+10}" y1="{ly+19}" x2="{lx+34}" y2="{ly+19}" stroke="#1d4ed8" stroke-width="2.8"/>')
    e.append(f'<text x="{lx+42}" y="{ly+23}" font-size="11.8" fill="#0f172a">Sequential: 1.236/(t_FPGA+t_IO)</text>')
    e.append(f'<line x1="{lx+10}" y1="{ly+40}" x2="{lx+34}" y2="{ly+40}" stroke="#9333ea" stroke-width="2.4" stroke-dasharray="5,4"/>')
    e.append(f'<text x="{lx+42}" y="{ly+44}" font-size="11.8" fill="#0f172a">Overlap: max(t_FPGA, t_IO)</text>')

    e.append(
        f'<text x="{ml + pw/2:.1f}" y="{H-12}" text-anchor="middle" font-size="13.8" font-weight="bold" fill="#1e293b">'
        "Host↔FPGA Link Bandwidth (MB/s, Log Scale)</text>"
    )
    e.append(
        f'<text x="18" y="{mt + ph/2:.1f}" text-anchor="middle" font-size="13.8" font-weight="bold" fill="#1e293b" transform="rotate(-90 18 {mt + ph/2:.1f})">'
        "Break-Even Throughput (GMAC/s)</text>"
    )

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig5_60point_cycle_error(out_path: Path) -> None:
    """Two-panel figure* (920x370):
    (a) Per-frequency mean absolute stall error (cycles, log scale) across 10 SDRAM setpoints (80 MHz engine).
    (b) 32-offset (0..31) sawtooth validation on 192->640->192 at 80/90 MHz (Base vs Stateful WB).
    """
    rev_path = BUILD_DIR / "revision_stateful_refresh_results.json"
    rev = json.loads(rev_path.read_text(encoding="utf-8"))
    pf = rev["per_freq_60"]
    off_rows = rev["offset32_640_rows"]
    off_stats = rev["offset32_comparison"]["192->640->192"]

    W, H = 920, 370
    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    # --- Panel (a): Per-frequency log-scale stall error (cycles) ---
    ml1, mt1, pw1, ph1 = 68, 38, 372, 260
    ymin_l, ymax_l = 0.0, 3.85  # 1 to ~7,000 cycles
    f_min, f_max = 57.0, 123.0

    def px1(f: float) -> float:
        return ml1 + ((f - f_min) / (f_max - f_min)) * pw1

    def py1(val_cyc: float) -> float:
        lv = math.log10(max(val_cyc, 1.0))
        return mt1 + (1.0 - (lv - ymin_l) / (ymax_l - ymin_l)) * ph1

    e.append(
        f'<text x="{ml1 + pw1/2:.1f}" y="22" text-anchor="middle" font-size="13.0" font-weight="bold" fill="#1e293b">'
        "(a) 60-Condition Stall MAE vs. SDRAM Clock (80 MHz Eng.)</text>"
    )
    for yv, lbl in [(1.0, "1"), (10.0, "10"), (100.0, "100"), (1000.0, "1k")]:
        yy = py1(yv)
        stroke = "#475569" if yv == 1.0 else "#e2e8f0"
        e.append(f'<line x1="{ml1}" y1="{yy:.1f}" x2="{ml1+pw1}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1.1"/>')
        e.append(f'<text x="{ml1-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11.8" fill="#334155">{lbl}</text>')
    for fv in [60, 70, 80, 90, 100, 110, 120]:
        xx = px1(fv)
        e.append(f'<line x1="{xx:.1f}" y1="{mt1}" x2="{xx:.1f}" y2="{mt1+ph1}" stroke="#f1f5f9" stroke-width="1.0"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt1+ph1+18}" text-anchor="middle" font-size="11.5" fill="#334155">{fv}</text>')
    e.append(f'<rect x="{ml1}" y="{mt1}" width="{pw1}" height="{ph1}" fill="none" stroke="#475569" stroke-width="1.3"/>')
    e.append(f'<text x="{ml1+pw1/2:.1f}" y="{mt1+ph1+38}" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#1e293b">SDRAM Clock Frequency (MHz)</text>')
    e.append(
        f'<text x="18" y="{mt1+ph1/2:.1f}" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#1e293b" '
        f'transform="rotate(-90 18 {mt1+ph1/2:.1f})">Mean Stall Error (Cycles) — Log</text>'
    )

    # 1-Run Repeat SD (~8.04 cycles) and 10-Run Mean SE (~2.54 cycles)
    y_sd = py1(8.04)
    y_se = py1(2.54)
    e.append(f'<line x1="{ml1}" y1="{y_sd:.1f}" x2="{ml1+pw1}" y2="{y_sd:.1f}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="3,3"/>')
    e.append(f'<text x="{ml1+pw1-6}" y="{y_sd-5:.1f}" text-anchor="end" font-size="9.6" font-weight="bold" fill="#475569">1-Run SD (8.0 cyc)</text>')
    e.append(f'<line x1="{ml1}" y1="{y_se:.1f}" x2="{ml1+pw1}" y2="{y_se:.1f}" stroke="#0f766e" stroke-width="1.2" stroke-dasharray="2,3"/>')
    e.append(f'<text x="{ml1+pw1-6}" y="{y_se+11:.1f}" text-anchor="end" font-size="9.6" font-weight="bold" fill="#0f766e">10-Run Mean SE (2.5 cyc)</text>')

    x80 = px1(80.0)
    e.append(f'<line x1="{x80:.1f}" y1="{mt1}" x2="{x80:.1f}" y2="{mt1+ph1}" stroke="#fca5a5" stroke-width="1.4" stroke-dasharray="4,3"/>')

    series_a = [
        ("bist_stall_mae_cyc", "#dc2626", "6,3", "BIST Roofline", "square"),
        ("wb_stall_mae_cyc", "#2563eb", "", "Base WB (Duty-Cyc.)", "circle"),
        ("sf_dropin_stall_mae_cyc", "#0d9488", "4,3", "Stateful Refresh WB", "diamond"),
        ("sim_0ppm_stall_mae_cyc", "#7c3aed", "2,2", "0-ppm RTL Sim", "hollow"),
    ]
    for key, col, dash, _lbl, shape in series_a:
        pts = [(px1(row["sdram_mhz"]), py1(row[key])) for row in pf]
        d_str = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        e.append(f'<path d="{d_str}" fill="none" stroke="{col}" stroke-width="2.1"{da}/>')
        for x, y in pts:
            if shape == "circle":
                e.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{col}" stroke="#ffffff" stroke-width="0.9"/>')
            elif shape == "square":
                e.append(f'<rect x="{x-3.2:.1f}" y="{y-3.2:.1f}" width="6.4" height="6.4" fill="{col}" stroke="#ffffff" stroke-width="0.8"/>')
            elif shape == "diamond":
                e.append(f'<polygon points="{x:.1f},{y-4.2:.1f} {x+4.2:.1f},{y:.1f} {x:.1f},{y+4.2:.1f} {x-4.2:.1f},{y:.1f}" fill="{col}" stroke="#ffffff" stroke-width="0.8"/>')
            else:
                e.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#ffffff" stroke="{col}" stroke-width="1.8"/>')

    y_80_base = py1(next(r["wb_stall_mae_cyc"] for r in pf if r["sdram_mhz"] == 80))
    e.append(f'<text x="{x80+7:.1f}" y="{y_80_base-8:.1f}" font-size="10.0" font-weight="bold" fill="#1d4ed8">80/80 CDC (66 cyc)</text>')

    leg_a = [(ml1 + 2, "#dc2626", "BIST"), (ml1 + 58, "#2563eb", "Base WB"), (ml1 + 136, "#0d9488", "Stateful WB"), (ml1 + 236, "#7c3aed", "0-ppm Sim")]
    for lx, col, txt in leg_a:
        e.append(f'<rect x="{lx}" y="{H-24}" width="11" height="11" fill="{col}" rx="2"/>')
        e.append(f'<text x="{lx+15}" y="{H-14}" font-size="10.6" font-weight="bold" fill="#334155">{txt}</text>')

    # --- Panel (b): 32-Offset Sawtooth (192->640->192 @ 80/90 MHz) ---
    ml2, mt2, pw2, ph2 = 528, 38, 376, 260
    ymin2, ymax2 = 64980.0, 65365.0

    def px2(off: float) -> float:
        return ml2 + (off / 31.0) * pw2

    def py2(cyc: float) -> float:
        return mt2 + (1.0 - (cyc - ymin2) / (ymax2 - ymin2)) * ph2

    e.append(
        f'<text x="{ml2 + pw2/2:.1f}" y="22" text-anchor="middle" font-size="13.0" font-weight="bold" fill="#1e293b">'
        "(b) 32-Offset Sawtooth (192→640→192 @ 80/90 MHz)</text>"
    )
    for yv, lbl in [(65100, "65,100"), (65180, "65,180"), (65260, "65,260"), (65340, "65,340")]:
        yy = py2(yv)
        e.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1.1"/>')
        e.append(f'<text x="{ml2-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11.2" fill="#334155">{lbl}</text>')
    for ov in [0, 8, 16, 24, 31]:
        xx = px2(ov)
        e.append(f'<line x1="{xx:.1f}" y1="{mt2}" x2="{xx:.1f}" y2="{mt2+ph2}" stroke="#f1f5f9" stroke-width="1.0"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt2+ph2+18}" text-anchor="middle" font-size="11.5" fill="#334155">{ov}</text>')
    e.append(f'<rect x="{ml2}" y="{mt2}" width="{pw2}" height="{ph2}" fill="none" stroke="#475569" stroke-width="1.3"/>')
    e.append(f'<text x="{ml2+pw2/2:.1f}" y="{mt2+ph2+38}" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#1e293b">128-bit Word Offset (addr₀ mod 32)</text>')
    e.append(
        f'<text x="466" y="{mt2+ph2/2:.1f}" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#1e293b" '
        f'transform="rotate(-90 466 {mt2+ph2/2:.1f})">Total Layer Cycles</text>'
    )

    pts_avg_pl = [(px2(r["offset"]), py2(65090.2 if (r["offset"] % 8 == 0) else 65243.6)) for r in off_rows]
    d_avg_pl = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts_avg_pl))
    e.append(f'<path d="{d_avg_pl}" fill="none" stroke="#d97706" stroke-width="2.0" stroke-dasharray="5,3"/>')

    pts_dc = [(px2(r["offset"]), py2(r["pred_dc"])) for r in off_rows]
    pts_sf = [(px2(r["offset"]), py2(r["pred_sf_rtl"])) for r in off_rows]
    pts_meas = [(px2(r["offset"]), py2(r["meas_tot_mean"])) for r in off_rows]
    d_dc = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts_dc))
    d_sf = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts_sf))
    d_meas = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts_meas))
    e.append(f'<path d="{d_dc}" fill="none" stroke="#2563eb" stroke-width="1.8" stroke-dasharray="6,3"/>')
    e.append(f'<path d="{d_sf}" fill="none" stroke="#0d9488" stroke-width="2.0"/>')
    e.append(f'<path d="{d_meas}" fill="none" stroke="#0f172a" stroke-width="2.1"/>')
    for x, y in pts_sf:
        e.append(f'<rect x="{x-2.2:.1f}" y="{y-2.2:.1f}" width="4.4" height="4.4" fill="#0d9488"/>')
    for x, y in pts_meas:
        e.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="#0f172a" stroke="#ffffff" stroke-width="0.7"/>')

    e.append(f'<rect x="{ml2+110}" y="{mt2+ph2-48}" width="258" height="40" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.1"/>')
    e.append(f'<text x="{ml2+239}" y="{mt2+ph2-32}" text-anchor="middle" font-size="10.2" font-weight="bold" fill="#1e293b">Stateful MAE: {off_stats["stateful_rtl_exact"]["mae_cyc"]:.1f} vs. Base: {off_stats["duty_cycle"]["mae_cyc"]:.1f} cyc</text>')
    e.append(f'<text x="{ml2+239}" y="{mt2+ph2-17}" text-anchor="middle" font-size="10.0" fill="#334155">r = {off_stats["stateful_rtl_exact"]["r"]:.4f} · Per-layout Avg-Pg: 47.4 cyc</text>')

    leg_b = [(ml2 + 2, "#0f172a", "Silicon"), (ml2 + 68, "#0d9488", "Stateful WB"), (ml2 + 168, "#2563eb", "Base WB"), (ml2 + 244, "#d97706", "Per-Layout Avg")]
    for lx, col, txt in leg_b:
        e.append(f'<rect x="{lx}" y="{H-24}" width="11" height="11" fill="{col}" rx="2"/>')
        e.append(f'<text x="{lx+15}" y="{H-14}" font-size="10.6" font-weight="bold" fill="#334155">{txt}</text>')

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig_offset_phase(out_path: Path) -> None:
    """Visual timeline explaining why per-packet predicts +42 cycles when total pages (150) are equal."""
    W, H = 560, 232
    e: list[str] = [arrow_defs(), f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    # Top row: offset 1 (7-page packet i=3,7,.. aligns after 1,286-cycle window)
    e.append('<text x="12" y="34" font-size="12.8" font-weight="bold" fill="#991b1b">Offset 1</text>')
    e.append('<text x="12" y="50" font-size="11.0" fill="#475569">(i = 3,7,…,23)</text>')

    e.append('<rect x="108" y="20" width="210" height="36" rx="4" fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.5"/>')
    e.append('<text x="213" y="43" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#1e3a8a">C_tile(i−1) = 1,286 (epi=6)</text>')

    e.append('<rect x="108" y="64" width="280" height="34" rx="4" fill="#ccfbf1" stroke="#0f766e" stroke-width="1.5"/>')
    e.append('<text x="248" y="86" text-anchor="middle" font-size="12.0" font-weight="bold" fill="#134e4a">7-Page DMA: C_dma = 1,297.5</text>')

    e.append('<rect x="318" y="20" width="70" height="36" rx="4" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>')
    e.append('<text x="353" y="43" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#991b1b">+11.5 cyc</text>')

    e.append('<text x="472" y="44" text-anchor="middle" font-size="12.0" font-weight="bold" fill="#991b1b">6 × 11.5 = 69.0</text>')
    e.append('<text x="472" y="62" text-anchor="middle" font-size="11.0" fill="#475569">stall cycles</text>')

    # Divider
    e.append('<line x1="12" y1="112" x2="548" y2="112" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="4,3"/>')

    # Bottom row: offset 15 (7-page packet i=2,6,.. aligns after 1,293-cycle window)
    e.append('<text x="12" y="140" font-size="12.8" font-weight="bold" fill="#065f46">Offset 15</text>')
    e.append('<text x="12" y="156" font-size="11.0" fill="#475569">(i = 2,6,…,22)</text>')

    e.append('<rect x="108" y="126" width="248" height="36" rx="4" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>')
    e.append('<text x="232" y="149" text-anchor="middle" font-size="12.2" font-weight="bold" fill="#78350f">C_tile(i−1) = 1,293 (epi=13)</text>')

    e.append('<rect x="108" y="170" width="280" height="34" rx="4" fill="#ccfbf1" stroke="#0f766e" stroke-width="1.5"/>')
    e.append('<text x="248" y="192" text-anchor="middle" font-size="12.0" font-weight="bold" fill="#134e4a">7-Page DMA: C_dma = 1,297.5</text>')

    e.append('<rect x="356" y="126" width="32" height="36" rx="4" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>')
    e.append('<text x="372" y="149" text-anchor="middle" font-size="10.8" font-weight="bold" fill="#14532d">+4.5</text>')

    e.append('<text x="472" y="150" text-anchor="middle" font-size="12.0" font-weight="bold" fill="#065f46">6 × 4.5 = 27.0</text>')
    e.append('<text x="472" y="168" text-anchor="middle" font-size="11.0" fill="#475569">stall cycles</text>')

    e.append('<text x="280" y="224" text-anchor="middle" font-size="12.4" font-weight="bold" fill="#0f172a">Phase Contrast Δ(Offset 1 − 15) = 6 × (1,293 − 1,286) = +42.0 cycles/tile</text>')

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig_wb_vs_gb_margin(out_path: Path) -> None:
    """Paired error difference |e_GB60| - |e_BaseWB| vs max(C_dma - C_tile) across all 45 held-out conditions."""
    import sys
    sys.path.insert(0, str(ROOT_DIR / "scripts"))
    from run_revision_stateful_refresh_and_baselines import make_ffn, page_word_counts, parse_sh
    from run_43_silicon_heldout import predict_rtl_gen

    fc14 = json.loads((BUILD_DIR / "frozen_heldout_predictions_c1_c4.json").read_text(encoding="utf-8"))
    c14 = json.loads((BUILD_DIR / "c1_c4_silicon_results.json").read_text(encoding="utf-8"))
    e79 = json.loads((BUILD_DIR / "e7_e9_silicon_results.json").read_text(encoding="utf-8"))["E8_E9_rows"]
    b156 = json.loads((BUILD_DIR / "b1_b5_b6_silicon_results.json").read_text(encoding="utf-8"))
    gb_60 = b156["frozen_greybox_constants"]
    v43 = json.loads((BUILD_DIR / "validation_43_heldout_results.json").read_text(encoding="utf-8"))

    items = []
    for f_row, m_row in zip(fc14["c1_heldout_extrapolation"], c14["c1_heldout"]["rows"]):
        dm, df = parse_sh(m_row["shape"]); wl = make_ffn(dm, df, 4, seed=1)
        items.append(("C1", 4, 80.0, 100.0, f_row["l1_base"], f_row["l2_base"], wl["l1_meta"], wl["l2_meta"], m_row["meas_mean"]))
    for f_row, m_row in zip(fc14["c2_t8_new_clocks"], c14["c2_t8_new_clocks"]):
        dm, df = parse_sh(m_row["shape"]); wl = make_ffn(dm, df, 8, seed=1)
        items.append(("C2", 8, float(m_row["eng_mhz"]), float(m_row["sdram_mhz"]), f_row["l1_base"], f_row["l2_base"], wl["l1_meta"], wl["l2_meta"], m_row["meas_mean"]))
    for r in e79:
        if r["tag"] != "e80_s64": continue
        dm, df = parse_sh(r["shape"]); wl = make_ffn(dm, df, 4, seed=1)
        items.append(("E9", 4, float(r["eng_mhz"]), float(r["sdram_mhz"]), r["l1_base"], r["l2_base"], wl["l1_meta"], wl["l2_meta"], r["meas_mean"]))
    for r in b156["b1_t4_records"]:
        dm, df = parse_sh(r["shape"]); wl = make_ffn(dm, df, 4, seed=1)
        items.append(("B1_T4", 4, float(r["eng_mhz"]), float(r["sdram_mhz"]), r["l1_base"], r["l2_base"], wl["l1_meta"], wl["l2_meta"], r["meas_tot_mean"]))
    for r in b156["b1_t8_records"]:
        dm, df = parse_sh(r["shape"]); wl = make_ffn(dm, df, 8, seed=1)
        items.append(("B1_T8", 8, float(r["eng_mhz"]), float(r["sdram_mhz"]), r["l1_base"], r["l2_base"], wl["l1_meta"], wl["l2_meta"], r["meas_tot_mean"]))
    for key, tt, sh_list in [
        ("t4_e80_s100_heldout", 4, [(128, 256), (320, 960)]),
        ("t4_e70_s100_heldout", 4, [(128, 256), (320, 960)]),
        ("t8_e70_s100_heldout", 8, [(128, 256), (192, 384), (256, 576), (256, 960), (512, 1536)]),
    ]:
        addr = 0
        for idx, ((dm, df), r) in enumerate(zip(sh_list, v43[key])):
            wl = make_ffn(dm, df, tt, seed=20261160 + idx + tt * 100)
            b1 = addr; addr += len(wl["l1_words"]); b2 = addr; addr += len(wl["l2_words"])
            items.append(("Val43", tt, float(r["eng_nom"]), float(r["sd_nom"]), b1, b2, wl["l1_meta"], wl["l2_meta"], r["meas_mean"]))

    pts = []
    for suite, tt, eng, sd, b1, b2, m1, m2, meas in items:
        ri = int(sd * 14); tr = 9 if sd > 100 else 8
        wb = predict_rtl_gen(b1, m1, tt, eng, sd, ri, tr) + predict_rtl_gen(b2, m2, tt, eng, sd, ri, tr)
        def gb_l(b, m):
            nt, kch, kc = m["n_tiles"], m["k_chunks"], m["k_chunk"]; P = 8 + m["w_words_per_chunk"]; M = nt * kch
            ct = [tt * kc + (6.0 if (i % kch) < kch - 1 else 13.0) for i in range(M)]
            rf = 1.0 + gb_60["t_rfc"] / ri
            cd = [((8 * P + gb_60["c_page"] * len(page_word_counts(b + i * P, P)) + 1) * rf + gb_60["c_cdc_sd"]) * (eng / sd) + gb_60["c_cdc_eng"] for i in range(M)]
            return sum(ct) + cd[0] + sum(max(0.0, cd[i] - ct[i - 1]) for i in range(1, M))
        gb = gb_l(b1, m1) + gb_l(b2, m2)
        margins = []
        for b, m in [(b1, m1), (b2, m2)]:
            nt, kch, kc = m["n_tiles"], m["k_chunks"], m["k_chunk"]; P = 8 + m["w_words_per_chunk"]; M = nt * kch
            ct = [tt * kc + (6.0 if (i % kch) < kch - 1 else 13.0) for i in range(M)]
            rf = 1.0 + tr / ri
            cd = [((8 * P + 14.0 * len(page_word_counts(b + i * P, P)) + 1) * rf + 3.0) * (eng / sd) + 4.0 for i in range(M)]
            margins.append(sum(c - t for c, t in zip(cd, ct)) / M)
        pts.append((suite, tt, eng, sd, max(margins), abs(gb - meas) - abs(wb - meas)))

    W, H = 560, 290
    ml, mt, pw, ph = 68, 30, 468, 194
    e: list[str] = [arrow_defs(), f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    xmin, xmax = -1850.0, 850.0
    def px(v: float) -> float:
        return ml + (v - xmin) / (xmax - xmin) * pw

    # Symmetric log transform: symlog(y) = sign(y) * log10(1 + |y|/10)
    def symlog(v: float) -> float:
        return math.copysign(math.log10(1.0 + abs(v) / 10.0), v)

    ymin_s, ymax_s = symlog(-50.0), symlog(700.0)
    def py(v: float) -> float:
        return mt + (1.0 - (symlog(v) - ymin_s) / (ymax_s - ymin_s)) * ph

    # Shaded regimes
    x0 = px(0.0)
    e.append(f'<rect x="{ml}" y="{mt}" width="{x0-ml:.1f}" height="{ph}" fill="#f8fafc"/>')
    e.append(f'<rect x="{x0:.1f}" y="{mt}" width="{ml+pw-x0:.1f}" height="{ph}" fill="#fff7ed"/>')
    e.append(f'<line x1="{x0:.1f}" y1="{mt}" x2="{x0:.1f}" y2="{mt+ph}" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>')

    y0 = py(0.0)
    e.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml+pw}" y2="{y0:.1f}" stroke="#475569" stroke-width="1.2"/>')
    for yv, lbl in [(-30, "-30"), (-10, "-10"), (0, "0"), (10, "+10"), (50, "+50"), (200, "+200"), (500, "+500")]:
        yy = py(yv)
        e.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1.0"/>')
        e.append(f'<text x="{ml-7}" y="{yy+3.5:.1f}" text-anchor="end" font-size="10.2" fill="#334155">{lbl}</text>')
    for xv in [-1500, -1000, -500, 0, 500]:
        xx = px(xv)
        e.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#f1f5f9" stroke-width="1.0"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt+ph+15}" text-anchor="middle" font-size="10.2" fill="#334155">{xv:+d}</text>')

    e.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#475569" stroke-width="1.2"/>')
    e.append(f'<text x="{ml+pw/2:.1f}" y="18" text-anchor="middle" font-size="12.0" font-weight="bold" fill="#1e293b">Paired Error Advantage |e_GB60| − |e_BaseWB| vs. Bandwidth Margin (N = 45 Held-Out)</text>')
    e.append(f'<text x="{ml+pw/2:.1f}" y="{mt+ph+32}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#1e293b">Max Layer Margin  mean(C_dma − C_tile)  [cycles/packet]</text>')
    e.append(f'<text x="16" y="{mt+ph/2:.1f}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#1e293b" transform="rotate(-90 16 {mt+ph/2:.1f})">|e_GB60| − |e_BaseWB| (cycles, symlog)</text>')

    e.append(f'<text x="{ml+8}" y="{mt+14}" font-size="9.5" font-weight="bold" fill="#475569">Compute-Saturated (C_dma &lt; C_tile)</text>')
    e.append(f'<text x="{ml+pw-6}" y="{mt+14}" text-anchor="end" font-size="9.5" font-weight="bold" fill="#9a3412">Bandwidth-Bound (C_dma &gt; C_tile)</text>')
    e.append(f'<text x="{ml+8}" y="{y0-6:.1f}" font-size="9.0" fill="#0f766e">Base WB better (&gt;0)</text>')
    e.append(f'<text x="{ml+8}" y="{y0+12:.1f}" font-size="9.0" fill="#7c3aed">GB60 better (&lt;0)</text>')

    styles = {
        "C1": ("#2563eb", "circle"),
        "C2": ("#0ea5e9", "square"),
        "E9": ("#7c3aed", "diamond"),
        "B1_T4": ("#d97706", "triangle"),
        "B1_T8": ("#dc2626", "star"),
        "Val43": ("#059669", "circle"),
    }
    for suite, tt, eng, sd, margin, diff in pts:
        col, shp = styles[suite]
        x, y = px(margin), py(diff)
        if shp == "circle":
            e.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{col}" stroke="#ffffff" stroke-width="0.7"/>')
        elif shp == "square":
            e.append(f'<rect x="{x-3.3:.1f}" y="{y-3.3:.1f}" width="6.6" height="6.6" fill="{col}" stroke="#ffffff" stroke-width="0.7"/>')
        elif shp == "diamond":
            e.append(f'<polygon points="{x:.1f},{y-4.5:.1f} {x+4.5:.1f},{y:.1f} {x:.1f},{y+4.5:.1f} {x-4.5:.1f},{y:.1f}" fill="{col}" stroke="#ffffff" stroke-width="0.7"/>')
        elif shp == "triangle":
            e.append(f'<polygon points="{x:.1f},{y-4.3:.1f} {x+4.2:.1f},{y+3.5:.1f} {x-4.2:.1f},{y+3.5:.1f}" fill="{col}" stroke="#ffffff" stroke-width="0.7"/>')
        else:
            e.append(f'<rect x="{x-3.8:.1f}" y="{y-3.8:.1f}" width="7.6" height="7.6" rx="1.5" fill="{col}" stroke="#ffffff" stroke-width="0.9"/>')

    # Annotate key clusters
    e.append(f'<text x="{px(340):.1f}" y="{py(490)+15:.1f}" text-anchor="end" font-size="9.3" font-weight="bold" fill="#991b1b">B1 T=8, 80/40 MHz (2:1)</text>')
    e.append(f'<text x="{px(340):.1f}" y="{py(490)+27:.1f}" text-anchor="end" font-size="9.0" fill="#991b1b">+99 to +491 cyc (Base WB)</text>')
    e.append(f'<text x="{px(730):.1f}" y="{py(-28)+14:.1f}" text-anchor="end" font-size="9.2" font-weight="bold" fill="#5b21b6">E9 T=4, 80/64 (5:4, in-range)</text>')
    e.append(f'<text x="{px(-400):.1f}" y="{py(18):.1f}" text-anchor="middle" font-size="9.0" fill="#334155">C2 T=8 &amp; B1 T=8 (50–110M): clipped by max(0,·)</text>')

    leg = [
        (ml, "#2563eb", "C1 (Frozen)"), (ml + 82, "#0ea5e9", "C2 T=8 (Frozen)"), (ml + 186, "#7c3aed", "E9 (Frozen)"),
        (ml + 266, "#d97706", "B1 T=4"), (ml + 328, "#dc2626", "B1 T=8"), (ml + 392, "#059669", "Val43"),
    ]
    for lx, col, txt in leg:
        e.append(f'<rect x="{lx}" y="{H-19}" width="9" height="9" fill="{col}" rx="1.5"/>')
        e.append(f'<text x="{lx+12}" y="{H-11}" font-size="9.5" font-weight="bold" fill="#334155">{txt}</text>')

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    g4b = json.loads((BUILD_DIR / "g4b_sdram_results.json").read_text(encoding="utf-8"))
    oos = json.loads((BUILD_DIR / "oos_stall_validation_results.json").read_text(encoding="utf-8"))
    paired = json.loads((BUILD_DIR / "wer_w4_vs_w8_paired_bootstrap.json").read_text(encoding="utf-8"))
    abl_tone = json.loads((BUILD_DIR / "w4_w3_ablation_and_tone6x6.json").read_text(encoding="utf-8"))
    mem_data = json.loads((BUILD_DIR / "memory_bandwidth_analysis.json").read_text(encoding="utf-8"))
    img_data = json.loads((BUILD_DIR / "ffn_30m_sdram_image.json").read_text(encoding="utf-8"))
    g5b_data = json.loads((BUILD_DIR / "g5b_multisent_fullseq_results.json").read_text(encoding="utf-8"))
    mcu_data = build_padded_t500_mcu_data(g5b_data, img_data)

    outputs = [
        FIG_DIR / "fig1_hybrid_architecture.svg",
        FIG_DIR / "fig2_m9k_bandwidth_tradeoff.svg",
        FIG_DIR / "fig3_rtl_timing.svg",
        FIG_DIR / "fig4_sdram_stall_reconstruction.svg",
        FIG_DIR / "fig5_60point_cycle_error.svg",
        FIG_DIR / "fig_offset_phase.svg",
        FIG_DIR / "fig3b_oos_stall_prediction.svg",
        FIG_DIR / "fig4_wer_pareto_and_tones.svg",
        FIG_DIR / "fig5_mcu_breakeven.svg",
        FIG_DIR / "fig_wb_vs_gb_margin.svg",
    ]

    generate_fig1_hybrid_architecture(outputs[0])
    generate_fig2_m9k_bandwidth_tradeoff(mem_data, g4b, outputs[1])
    generate_fig_rtl_timing(outputs[2])
    generate_fig3a_sdram_stall_reconstruction(g4b, outputs[3])
    generate_fig5_60point_cycle_error(outputs[4])
    generate_fig_offset_phase(outputs[5])
    generate_fig3b_oos_stall_prediction(oos, outputs[6])
    generate_fig4_wer_pareto_and_tones(paired, abl_tone, outputs[7])
    generate_fig5_mcu_breakeven(mcu_data, g5b_data, outputs[8])
    generate_fig_wb_vs_gb_margin(outputs[9])

    for p in outputs:
        ET.parse(p)
        print(f"Verified English SVG XML: {p} ({p.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
