#!/usr/bin/env python3
"""Generate publication-grade vector SVG figures for the official Q2 journal manuscript
(pure Python stdlib, IEEE 2-column ready, Vietnamese decimal comma formatting).

Outputs in docs/figures/:
  1. fig1_hybrid_architecture.svg         (Closed-loop Host <-> EP4CE6 <-> SDRAM architecture)
  2. fig2_m9k_bandwidth_tradeoff.svg      (M9K BRAM budget & decimal MB/s vs T_tile)
  3. fig3a_sdram_stall_reconstruction.svg (5-freq 512->1536->512 HW-counter stall reconstruction)
  4. fig3b_oos_stall_prediction.svg       (Locked RTL out-of-sample stall prediction on 3 shapes)
  5. fig4_wer_pareto_and_tones.svg        (Delta-WER 19-speaker CI & 6-tone error comparison)
  6. fig5_mcu_breakeven.svg               (FFN-only embedded host break-even vs bus bandwidth)
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
FIG_DIR = ROOT_DIR / "docs" / "figures"

FONT_FAMILY = "DejaVu Sans, Arial, Helvetica, sans-serif"


def xml_esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_vi(val: float, digits: int = 2, signed: bool = False) -> str:
    if signed:
        s = f"{val:+.{digits}f}"
    else:
        s = f"{val:.{digits}f}"
    return s.replace(".", ",")


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
        '    <marker id="arr_blue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
        '      <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#1d4ed8"/>\n'
        "    </marker>\n"
        '    <marker id="arr_orange" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
        '      <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#c2410c"/>\n'
        "    </marker>\n"
        '    <marker id="arr_teal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
        '      <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#0f766e"/>\n'
        "    </marker>\n"
        '    <marker id="arr_slate" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">\n'
        '      <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#475569"/>\n'
        "    </marker>\n"
        "  </defs>"
    )


def generate_fig1_hybrid_architecture(out_path: Path) -> None:
    W, H = 940, 472
    e: list[str] = [arrow_defs(), f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    # Title & subtitle
    e.append(
        f'<text x="{W/2:.1f}" y="24" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#0f172a">'
        "Hình 1. Kiến trúc Đồng thiết kế Hybrid Vòng kín Host ↔ Cyclone IV EP4CE10E22C8N ↔ SDRAM Ngoài 8 MiB</text>"
    )
    e.append(
        f'<text x="{W/2:.1f}" y="42" text-anchor="middle" font-size="10.8" fill="#475569">'
        "Phân vùng theo năng lực silicon: 57,92% MAC &amp; Decoder/Joiner ở Host · 36 khối FFN (42,08% MAC) thường trú trong SDRAM ngoài trên board</text>"
    )

    # Panel 1: HOST (x=18..252, y=62..392)
    e.append('<rect x="18" y="62" width="234" height="334" rx="8" fill="#f8fafc" stroke="#334155" stroke-width="1.6"/>')
    e.append('<rect x="18" y="62" width="234" height="34" rx="8" fill="#1e293b"/>')
    e.append('<rect x="18" y="84" width="234" height="12" fill="#1e293b"/>')
    e.append('<text x="135" y="84" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#ffffff">HOST (37 Phân đoạn ONNX)</text>')

    host_boxes = [
        (108, 42, "#e2e8f0", "#475569", "Âm thanh 16 kHz → Fbank 80-bin", "Độ dài T_fbank bất kỳ"),
        (164, 56, "#dbeafe", "#1d4ed8", "Embed + Conv + Attention", "1,702 GMAC (57,92%) @ T=500"),
        (234, 54, "#fef3c7", "#b45309", "Bộ đệm/cắt đuôi T_tile = 4", "Lấp 1–3 khung ở stage có mod 4 ≠ 0"),
        (304, 76, "#dcfce7", "#15803d", "RNN-T Decoder &amp; Joiner", "Đầu ra encoder khớp 0,0 · Transcript trùng ONNX"),
    ]
    for by, bh, fill, stroke, t1, t2 in host_boxes:
        e.append(f'<rect x="32" y="{by}" width="206" height="{bh}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        e.append(f'<text x="135" y="{by + bh*0.44:.1f}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#0f172a">{t1}</text>')
        e.append(f'<text x="135" y="{by + bh*0.77:.1f}" text-anchor="middle" font-size="9.6" fill="#334155">{t2}</text>')

    # Arrows inside Host
    e.append('<line x1="135" y1="150" x2="135" y2="162" stroke="#475569" stroke-width="1.5" marker-end="url(#arr_slate)"/>')
    e.append('<line x1="135" y1="220" x2="135" y2="232" stroke="#475569" stroke-width="1.5" marker-end="url(#arr_slate)"/>')
    e.append('<line x1="135" y1="288" x2="135" y2="302" stroke="#475569" stroke-width="1.5" marker-end="url(#arr_slate)"/>')

    # Bridge Host <-> FPGA (x=252..348)
    # Forward tensor arrow (Host -> FPGA)
    e.append('<line x1="252" y1="185" x2="344" y2="185" stroke="#1d4ed8" stroke-width="2.2" marker-end="url(#arr_blue)"/>')
    e.append('<rect x="257" y="138" width="84" height="40" rx="4" fill="#eff6ff" stroke="#93c5fd" stroke-width="1"/>')
    e.append('<text x="299" y="154" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#1e3a8a">x_q INT8</text>')
    e.append('<text x="299" y="169" text-anchor="middle" font-size="9.0" fill="#1e40af">53,6 KB/s ghi</text>')

    # Return tensor arrow (FPGA -> Host)
    e.append('<line x1="346" y1="262" x2="254" y2="262" stroke="#c2410c" stroke-width="2.2" marker-end="url(#arr_orange)"/>')
    e.append('<rect x="257" y="270" width="84" height="40" rx="4" fill="#fff7ed" stroke="#fdba74" stroke-width="1"/>')
    e.append('<text x="299" y="286" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#9a3412">y_out INT8</text>')
    e.append('<text x="299" y="301" text-anchor="middle" font-size="9.0" fill="#c2410c">5,76 KB/s đọc</text>')

    # One-time upload label above arrows
    e.append('<rect x="255" y="76" width="88" height="46" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
    e.append('<text x="299" y="92" text-anchor="middle" font-size="9.0" font-weight="bold" fill="#334155">Nạp 1 lần/phiên</text>')
    e.append('<text x="299" y="105" text-anchor="middle" font-size="8.8" fill="#475569">6,52 MiB (~216 s)</text>')
    e.append('<text x="299" y="117" text-anchor="middle" font-size="8.6" fill="#0f766e">0 B trọng số/câu</text>')

    # Target bus callout under arrows
    e.append('<rect x="255" y="324" width="88" height="54" rx="4" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1"/>')
    e.append('<text x="299" y="339" text-anchor="middle" font-size="8.8" font-weight="bold" fill="#065f46">Định hướng nhúng</text>')
    e.append('<text x="299" y="352" text-anchor="middle" font-size="8.6" fill="#047857">Bus 5–10 MB/s</text>')
    e.append('<text x="299" y="364" text-anchor="middle" font-size="8.5" fill="#047857">(FIFO / FT232H)</text>')
    e.append('<text x="299" y="375" text-anchor="middle" font-size="8.3" fill="#065f46">giảm nghẽn JTAG</text>')

    # Panel 2: FPGA EP4CE10 (x=348..658, y=62..396)
    e.append('<rect x="348" y="62" width="312" height="334" rx="8" fill="#f8fafc" stroke="#1d4ed8" stroke-width="1.8"/>')
    e.append('<rect x="348" y="62" width="312" height="36" rx="8" fill="#1d4ed8"/>')
    e.append('<rect x="348" y="86" width="312" height="12" fill="#1d4ed8"/>')
    e.append('<text x="504" y="81" text-anchor="middle" font-size="11.2" font-weight="bold" fill="#ffffff">FPGA Cyclone IV EP4CE10E22C8N (FFN W4A8)</text>')
    e.append('<text x="504" y="94" text-anchor="middle" font-size="8.4" fill="#dbeafe">6.056/10.320 LE (59%) · 26/46 M9K · 20/46 DSP · 190,65 mW FPGA-only</text>')

    # Top row inside FPGA: JTAG bridge & PLL
    e.append('<rect x="360" y="106" width="136" height="38" rx="4" fill="#e2e8f0" stroke="#64748b" stroke-width="1"/>')
    e.append('<text x="428" y="122" text-anchor="middle" font-size="9.4" font-weight="bold" fill="#0f172a">sld_virtual_jtag</text>')
    e.append('<text x="428" y="136" text-anchor="middle" font-size="7.8" fill="#475569">CDC 2-FF · STA nội bộ TNS=0</text>')

    e.append('<rect x="504" y="106" width="144" height="38" rx="4" fill="#e0e7ff" stroke="#4f46e5" stroke-width="1"/>')
    e.append('<text x="576" y="122" text-anchor="middle" font-size="9.4" font-weight="bold" fill="#312e81">altpll: 2 Miền Xung</text>')
    e.append('<text x="576" y="136" text-anchor="middle" font-size="8.4" fill="#3730a3">80 MHz Eng | 100 MHz SDRAM</text>')

    # Engine Datapath Box (x=360..648, y=154..284)
    e.append('<rect x="360" y="154" width="288" height="130" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.3"/>')
    e.append('<text x="504" y="170" text-anchor="middle" font-size="10.2" font-weight="bold" fill="#1e3a8a">Pipeline Tính FFN Khớp Bit 0 LSB (T_tile = 4)</text>')

    e.append('<rect x="370" y="180" width="84" height="54" rx="4" fill="#ffffff" stroke="#60a5fa" stroke-width="1"/>')
    e.append('<text x="412" y="197" text-anchor="middle" font-size="9.4" font-weight="bold" fill="#0f172a">act / h / y M9K</text>')
    e.append('<text x="412" y="211" text-anchor="middle" font-size="8.6" fill="#334155">12 M9K nội</text>')
    e.append('<text x="412" y="224" text-anchor="middle" font-size="8.6" fill="#334155">psum 64×32b</text>')

    e.append('<rect x="466" y="180" width="90" height="54" rx="4" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/>')
    e.append('<text x="511" y="196" text-anchor="middle" font-size="9.4" font-weight="bold" fill="#1e3a8a">16 Làn MAC</text>')
    e.append('<text x="511" y="210" text-anchor="middle" font-size="8.6" fill="#1e40af">Giải nén W4→I8</text>')
    e.append('<text x="511" y="224" text-anchor="middle" font-size="8.6" fill="#1e40af">K-tiling + cây cộng</text>')

    e.append('<rect x="568" y="180" width="70" height="54" rx="4" fill="#fef3c7" stroke="#d97706" stroke-width="1"/>')
    e.append('<text x="603" y="196" text-anchor="middle" font-size="9.3" font-weight="bold" fill="#92400e">Epilogue</text>')
    e.append('<text x="603" y="210" text-anchor="middle" font-size="8.5" fill="#78350f">Dyadic M,S</text>')
    e.append('<text x="603" y="224" text-anchor="middle" font-size="8.5" fill="#78350f">SwooshL LUT</text>')

    e.append('<line x1="454" y1="207" x2="464" y2="207" stroke="#1d4ed8" stroke-width="1.4" marker-end="url(#arr_blue)"/>')
    e.append('<line x1="556" y1="207" x2="566" y2="207" stroke="#1d4ed8" stroke-width="1.4" marker-end="url(#arr_blue)"/>')

    # Counters strip at bottom of engine box
    e.append('<rect x="370" y="244" width="268" height="30" rx="4" fill="#ffffff" stroke="#93c5fd" stroke-width="1"/>')
    e.append('<text x="504" y="262" text-anchor="middle" font-size="7.7" font-weight="bold" fill="#1e293b">'
             'Checksum + Bộ đếm 32-bit: C_tot, C_mac, C_stall, C_dma (free_clk: 64b)</text>')

    # Ping-Pong DMA & SDRAM Ctrl (y=296..384)
    e.append('<rect x="360" y="296" width="138" height="88" rx="5" fill="#f0fdfa" stroke="#0d9488" stroke-width="1.2"/>')
    e.append('<text x="429" y="315" text-anchor="middle" font-size="9.8" font-weight="bold" fill="#115e59">Ping-Pong DMA</text>')
    e.append('<text x="429" y="331" text-anchor="middle" font-size="8.8" fill="#134e4a">2 × 4 KiB Wgt + Param</text>')
    e.append('<text x="429" y="345" text-anchor="middle" font-size="8.8" fill="#134e4a">(8 M9K nạp trước W4)</text>')
    e.append('<text x="429" y="362" text-anchor="middle" font-size="8.6" font-weight="bold" fill="#0f766e">Gối đầu chunk ≥ 90 MHz</text>')
    e.append('<text x="429" y="376" text-anchor="middle" font-size="8.5" fill="#0f766e">Stall = 0,91% @ 100 MHz</text>')

    e.append('<rect x="510" y="296" width="138" height="88" rx="5" fill="#ecfeff" stroke="#0891b2" stroke-width="1.2"/>')
    e.append('<text x="579" y="315" text-anchor="middle" font-size="9.8" font-weight="bold" fill="#155e75">SDRAM Controller</text>')
    e.append('<text x="579" y="331" text-anchor="middle" font-size="8.8" fill="#164e63">FSM Page-Burst 256 cột</text>')
    e.append('<text x="579" y="345" text-anchor="middle" font-size="8.8" fill="#164e63">rd_lat = 4 · 16-bit SDR</text>')
    e.append('<text x="579" y="362" text-anchor="middle" font-size="8.6" font-weight="bold" fill="#0e7490">Đọc 188,53 MB/s (94,3%)</text>')
    e.append('<text x="579" y="376" text-anchor="middle" font-size="8.5" fill="#0e7490">Ghi 192,09 MB/s (96,0%)</text>')

    # Arrows DMA <-> MAC & DMA <-> SDRAM Ctrl
    e.append('<line x1="429" y1="296" x2="488" y2="236" stroke="#0f766e" stroke-width="1.6" marker-end="url(#arr_teal)"/>')
    e.append('<line x1="510" y1="340" x2="500" y2="340" stroke="#0f766e" stroke-width="1.6" marker-end="url(#arr_teal)"/>')

    # Bus SDRAM Ctrl <-> External SDRAM (x=660..726)
    e.append('<line x1="722" y1="322" x2="662" y2="322" stroke="#0f766e" stroke-width="2.6" marker-end="url(#arr_teal)"/>')
    e.append('<line x1="660" y1="352" x2="720" y2="352" stroke="#475569" stroke-width="1.6" marker-end="url(#arr_slate)"/>')
    e.append('<text x="692" y="308" text-anchor="middle" font-size="9.0" font-weight="bold" fill="#0f766e">16-bit SDR</text>')
    e.append('<text x="692" y="372" text-anchor="middle" font-size="8.8" fill="#334155">100 MHz</text>')

    # Panel 3: EXTERNAL SDRAM (x=724..922, y=62..396)
    e.append('<rect x="724" y="62" width="198" height="334" rx="8" fill="#f8fafc" stroke="#0f766e" stroke-width="1.8"/>')
    e.append('<rect x="724" y="62" width="198" height="36" rx="8" fill="#0f766e"/>')
    e.append('<rect x="724" y="86" width="198" height="12" fill="#0f766e"/>')
    e.append('<text x="823" y="81" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#ffffff">SDRAM Ngoài trên Board</text>')
    e.append('<text x="823" y="94" text-anchor="middle" font-size="9.0" fill="#ccfbf1">HY57V641620FTP-H · 8 MiB · 16-bit</text>')

    # Memory map bar (y=114..328)
    # 81.45% occupied (height 174 of 214) -> y=114..288
    e.append('<rect x="742" y="114" width="162" height="148" rx="4" fill="#99f6e4" stroke="#0f766e" stroke-width="1.2"/>')
    e.append('<text x="823" y="138" text-anchor="middle" font-size="10.0" font-weight="bold" fill="#134e4a">36 Khối FFN Thường trú</text>')
    e.append('<text x="823" y="156" text-anchor="middle" font-size="9.2" fill="#115e59">6,52 MiB (81,45% SDRAM)</text>')
    e.append('<text x="823" y="176" text-anchor="middle" font-size="8.8" fill="#134e4a">• Trọng số W4: 6,19 MiB</text>')
    e.append('<text x="823" y="193" text-anchor="middle" font-size="8.8" fill="#134e4a">• Header tham số: 336 KiB (lặp K)</text>')
    e.append('<text x="823" y="210" text-anchor="middle" font-size="8.8" fill="#134e4a">• Tổng 427.008 từ 128-bit</text>')
    e.append('<text x="823" y="232" text-anchor="middle" font-size="8.8" font-weight="bold" fill="#0f766e">Đọc lặp lại 24,22× @ T=500</text>')
    e.append('<text x="823" y="248" text-anchor="middle" font-size="8.6" fill="#115e59">157,8 MiB (165,4 MB)</text>')

    # Free space 18.55% -> y=268..318
    e.append('<rect x="742" y="268" width="162" height="48" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4,3"/>')
    e.append('<text x="823" y="289" text-anchor="middle" font-size="9.4" font-weight="bold" fill="#475569">Vùng trống: 1,48 MiB</text>')
    e.append('<text x="823" y="305" text-anchor="middle" font-size="8.6" fill="#64748b">(18,55% · không chứa đủ 16,99 MiB)</text>')

    e.append('<rect x="742" y="328" width="162" height="54" rx="4" fill="#ffffff" stroke="#0d9488" stroke-width="1"/>')
    e.append('<text x="823" y="346" text-anchor="middle" font-size="9.2" font-weight="bold" fill="#0f766e">BIST Toàn chip 8 MiB</text>')
    e.append('<text x="823" y="362" text-anchor="middle" font-size="8.8" fill="#1e293b">0 lỗi bit (60–120 MHz)</text>')
    e.append('<text x="823" y="376" text-anchor="middle" font-size="8.4" fill="#475569">4 bank × 4.096 row × 256 col</text>')

    # Bottom verification bar spanning full width (y=408..458)
    e.append('<rect x="18" y="408" width="904" height="50" rx="6" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.2"/>')
    e.append(
        '<text x="470" y="427" text-anchor="middle" font-size="10.0" font-weight="bold" fill="#14532d">'
        "Kiểm chứng từng byte: 2.448/2.448 block-tile y_out (2.374.656 B) + 144 tile stress h_post (417.792 B) = 2.792.448 B (0 lỗi, max_abs_diff = 0)</text>"
    )
    e.append(
        '<text x="470" y="446" text-anchor="middle" font-size="9.8" fill="#166534">'
        "Vòng kín Host↔FPGA (T=491, 4,906 s): FFN FPGA = 1,015 s · Wall-clock có kiểm chứng = 209,705 s · Đầu ra 0,0 · Transcript trùng ONNX 3/3 câu</text>"
    )

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig2_m9k_bandwidth_tradeoff(mem_data: dict, g4b: dict, out_path: Path) -> None:
    matrix = mem_data["tiling_feasibility_matrix"]
    lookup = {(row["arch"], row["t_tile"], row["w_bits"]): row for row in matrix}
    bist_100_rd = float(g4b["sdram_100mhz"]["bist_8mib"]["rd_MBps_1e6"])  # 188.53 MB/s

    W, H = 920, 440
    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    e.append(
        f'<text x="{W/2:.1f}" y="24" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#0f172a">'
        "Hình 2. Không gian Đánh đổi Tài nguyên M9K BRAM và Băng thông SDRAM Theo T_tile (512→1536→512 @ 80 MHz)</text>"
    )

    # ---------------- Panel (a): M9K BRAM Budget ----------------
    ml1, mt1, pw1, ph1 = 68, 72, 360, 276
    ymax1 = 72.0

    def py1(v: float) -> float:
        return mt1 + (1.0 - v / ymax1) * ph1

    e.append(
        f'<text x="{ml1 + pw1/2:.1f}" y="52" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e293b">'
        "(a) Số khối nhớ M9K trên chip (đơn vị: khối M9K)</text>"
    )
    for yv in [0, 10, 20, 30, 40, 50, 60, 70]:
        yy = py1(yv)
        stroke = "#cbd5e1" if yv == 0 else "#e2e8f0"
        e.append(f'<line x1="{ml1}" y1="{yy:.1f}" x2="{ml1+pw1}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"/>')
        e.append(f'<text x="{ml1-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.5" fill="#334155">{yv}</text>')
    e.append(f'<rect x="{ml1}" y="{mt1}" width="{pw1}" height="{ph1}" fill="none" stroke="#475569" stroke-width="1.2"/>')

    # Horizontal limits EP4CE6 (30), EP4CE10 (46), DE0-Nano (66)
    for cap, col, lbl, anc, lx_lbl, dy in [
        (30, "#dc2626", "EP4CE6 = 30 M9K", "end", ml1 + pw1 - 6, -5),
        (46, "#2563eb", "EP4CE10 = 46 M9K", "start", ml1 + 8, -5),
        (66, "#059669", "DE0-Nano = 66 M9K", "end", ml1 + pw1 - 6, -5),
    ]:
        yy = py1(cap)
        e.append(f'<line x1="{ml1}" y1="{yy:.1f}" x2="{ml1+pw1}" y2="{yy:.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="5,4"/>')
        e.append(f'<text x="{lx_lbl}" y="{yy+dy:.1f}" text-anchor="{anc}" font-size="9.5" font-weight="bold" fill="{col}">{lbl}</text>')

    # Groups for T=4, 8, 16: Std-Single, Std-Double(DMA), HF-Double(DMA)
    t_vals = [4, 8, 16]
    bar_w = 24.0
    gap = 6.0
    group_centers = [ml1 + 64, ml1 + 180, ml1 + 296]

    for idx, tv in enumerate(t_vals):
        cx = group_centers[idx]
        std = lookup[("standard_hpost_buf", tv, 4)]
        hf = lookup[("hidden_fused_y_psum", tv, 4)]
        bars = [
            (std["m9k_single_wgt"], "#93c5fd", "#1d4ed8"),
            (std["m9k_double_wgt_dma"], "#2563eb", "#1e3a8a"),
            (hf["m9k_double_wgt_dma"], "#f59e0b", "#b45309"),
        ]
        x0 = cx - (3 * bar_w + 2 * gap) / 2.0
        for bi, (val, bfill, bstroke) in enumerate(bars):
            bx = x0 + bi * (bar_w + gap)
            by = py1(val)
            bh = py1(0) - by
            e.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{bfill}" stroke="{bstroke}" stroke-width="1.1"/>')
            e.append(f'<text x="{bx + bar_w/2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#0f172a">{val}</text>')
        e.append(f'<text x="{cx:.1f}" y="{mt1+ph1+18}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#1e293b">T_tile = {tv}</text>')

    # Callout on selected T=4 Standard double DMA = 26 M9K
    e.append('<text x="132" y="216" text-anchor="middle" font-size="9.2" font-weight="bold" fill="#15803d">Chọn T=4 (26 ≤ 30 M9K)</text>')
    e.append('<text x="132" y="229" text-anchor="middle" font-size="8.8" fill="#166534">Đạt M9K &amp; BW (W4 DMA 2 đệm)</text>')

    # Legend (a)
    e.append(f'<rect x="{ml1+8}" y="{mt1+8}" width="174" height="58" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    for li, (col, text) in enumerate([("#93c5fd", "Standard (Single Wgt)"), ("#2563eb", "Standard (Double DMA)"), ("#f59e0b", "Hidden-Fused (Double DMA)")]):
        ly = mt1 + 16 + li * 16
        e.append(f'<rect x="{ml1+16}" y="{ly}" width="12" height="10" fill="{col}" stroke="#475569" stroke-width="0.6"/>')
        e.append(f'<text x="{ml1+34}" y="{ly+9}" font-size="9.2" fill="#0f172a">{text}</text>')

    e.append(f'<text x="20" y="{mt1+ph1/2:.1f}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#1e293b" transform="rotate(-90 20 {mt1+ph1/2:.1f})">Tài nguyên bộ nhớ khối (M9K)</text>')

    # ---------------- Panel (b): Decimal MB/s External BW ----------------
    ml2, mt2, pw2, ph2 = 508, 72, 384, 276
    ymax2 = 360.0

    def py2(v: float) -> float:
        return mt2 + (1.0 - v / ymax2) * ph2

    e.append(
        f'<text x="{ml2 + pw2/2:.1f}" y="52" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e293b">'
        "(b) Băng thông ngoài yêu cầu (MB/s thập phân, 10⁶ B/s)</text>"
    )
    for yv in [0, 50, 100, 150, 200, 250, 300, 350]:
        yy = py2(yv)
        stroke = "#cbd5e1" if yv == 0 else "#e2e8f0"
        e.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"/>')
        e.append(f'<text x="{ml2-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.5" fill="#334155">{yv}</text>')
    e.append(f'<rect x="{ml2}" y="{mt2}" width="{pw2}" height="{ph2}" fill="none" stroke="#475569" stroke-width="1.2"/>')

    # Theoretical peak 200 MB/s and BIST 188.53 MB/s
    y200 = py2(200.0)
    y188 = py2(bist_100_rd)
    e.append(f'<line x1="{ml2}" y1="{y200:.1f}" x2="{ml2+pw2}" y2="{y200:.1f}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="3,3"/>')
    e.append(f'<text x="{ml2+6}" y="{y200-5:.1f}" font-size="9.2" font-weight="bold" fill="#475569">Đỉnh lý thuyết: 200,0 MB/s</text>')
    e.append(f'<line x1="{ml2}" y1="{y188:.1f}" x2="{ml2+pw2}" y2="{y188:.1f}" stroke="#0f766e" stroke-width="1.6" stroke-dasharray="6,3"/>')
    e.append(f'<text x="{ml2+pw2-6}" y="{y188+12:.1f}" text-anchor="end" font-size="9.0" font-weight="bold" fill="#0f766e">BIST đọc: {fmt_vi(bist_100_rd, 2)} MB/s</text>')

    group_centers2 = [ml2 + 68, ml2 + 192, ml2 + 316]
    bar_w2 = 26.0
    gap2 = 6.0
    for idx, tv in enumerate(t_vals):
        cx = group_centers2[idx]
        # Compute decimal MB/s strictly from 10^6 bytes/s:
        # W4 pure = 640 / T ; W4 + 264/256 header = 660 / T ; W8 pure = 1280 / T
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
            e.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w2}" height="{bh:.1f}" fill="{bfill}" stroke="{bstroke}" stroke-width="1.1"/>')
            lbl = fmt_vi(val, 1 if val >= 80 or val == 40.0 else 2)
            if abs(val - 165.0) < 0.2:
                lbl = "165,0"
            elif abs(val - 160.0) < 0.2:
                lbl = "160,0"
            elif abs(val - 320.0) < 0.2:
                lbl = "320,0"
            elif abs(val - 82.5) < 0.2:
                lbl = "82,5"
            elif abs(val - 41.25) < 0.2:
                lbl = "41,25"
            e.append(f'<text x="{bx + bar_w2/2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="8.8" font-weight="bold" fill="#0f172a">{lbl}</text>')
        e.append(f'<text x="{cx:.1f}" y="{mt2+ph2+18}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#1e293b">T_tile = {tv}</text>')

    # Legend (b)
    e.append(f'<rect x="{ml2+pw2-186}" y="{mt2+8}" width="178" height="58" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    for li, (col, text) in enumerate([("#60a5fa", "W4 thuần (640 / T)"), ("#0d9488", "W4 + header (660 / T)"), ("#ef4444", "W8 thuần (1.280 / T)")]):
        ly = mt2 + 16 + li * 16
        e.append(f'<rect x="{ml2+pw2-178}" y="{ly}" width="12" height="10" fill="{col}" stroke="#475569" stroke-width="0.6"/>')
        e.append(f'<text x="{ml2+pw2-160}" y="{ly+9}" font-size="9.2" fill="#0f172a">{text}</text>')

    e.append(f'<text x="462" y="{mt2+ph2/2:.1f}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#1e293b" transform="rotate(-90 462 {mt2+ph2/2:.1f})">Băng thông ngoài yêu cầu (MB/s)</text>')

    # Bottom shared note
    e.append(
        f'<text x="{W/2:.1f}" y="{H-14}" text-anchor="middle" font-size="10.2" fill="#475569">'
        "Chú thích: Bộ nhớ M9K dùng chuẩn 1 KiB/M9K ở độ rộng 128-bit; băng thông dùng hệ thập phân MB/s (10⁶ B/s) thống nhất với BIST SDRAM 188,53 MB/s.</text>"
    )

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig3a_sdram_stall_reconstruction(g4b: dict, out_path: Path) -> None:
    """Plot 5-freq silicon stall vs hardware-counter reconstruction (explicitly labeled as reconstruction)."""
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

    W, H = 780, 456
    ml, mr, mt, mb = 74, 28, 62, 64
    pw, ph = W - ml - mr, H - mt - mb
    xmin, xmax = 55.0, 125.0
    ymin, ymax = 0.0, 40.0

    def px(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * pw

    def py(y: float) -> float:
        return mt + (1.0 - (y - ymin) / (ymax - ymin)) * ph

    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']

    x_crit = px(f_crit_mhz)
    e.append(f'<rect x="{ml:.1f}" y="{mt:.1f}" width="{x_crit - ml:.1f}" height="{ph:.1f}" fill="#fef2f2" opacity="0.72"/>')
    e.append(f'<rect x="{x_crit:.1f}" y="{mt:.1f}" width="{ml + pw - x_crit:.1f}" height="{ph:.1f}" fill="#f0fdf4" opacity="0.72"/>')

    for yv in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
        yy = py(yv)
        stroke = "#cbd5e1" if yv == 0 else "#e2e8f0"
        e.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"/>')
        e.append(f'<text x="{ml-10}" y="{yy+4:.1f}" text-anchor="end" font-size="11.2" fill="#334155">{yv}%</text>')

    for xv in [60, 70, 80, 90, 100, 110, 120]:
        xx = px(xv)
        e.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#e2e8f0" stroke-width="1"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt+ph+19}" text-anchor="middle" font-size="11.2" fill="#334155">{xv}</text>')

    e.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#475569" stroke-width="1.2"/>')

    e.append(f'<line x1="{x_crit:.1f}" y1="{mt}" x2="{x_crit:.1f}" y2="{mt+ph}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="6,4"/>')
    e.append(f'<text x="{x_crit+6:.1f}" y="{mt+19}" font-size="10.8" font-weight="bold" fill="#b91c1c">Ngưỡng chuyển pha ~{fmt_vi(f_crit_mhz, 1)} MHz (~164 MB/s)</text>')
    e.append(f'<text x="{px(69.5):.1f}" y="{mt+20}" text-anchor="middle" font-size="11.2" font-weight="bold" fill="#991b1b">Vùng nghẽn băng thông</text>')
    e.append(f'<text x="{px(108.5):.1f}" y="{mt+39}" text-anchor="middle" font-size="11.0" font-weight="bold" fill="#166534">Vùng bão hòa tính toán (chỉ còn trễ cold-start)</text>')

    curve_pts = [f"{px(55.0 + 0.5*k):.2f},{py(recon_curve_pct(55.0 + 0.5*k)):.2f}" for k in range(141)]
    e.append(f'<polyline points="{" ".join(curve_pts)}" fill="none" stroke="#1d4ed8" stroke-width="2.4"/>')

    # Carefully placed non-overlapping point annotations
    label_specs = {
        60: (144.0, 114.0, "start", False),
        80: (294.0, 286.0, "end", True),
        90: (418.0, 312.0, "start", True),
        100: (516.0, 272.0, "start", True),
        120: (736.0, 338.0, "end", True),
    }
    for p in meas_pts:
        cx, cy = px(p["sd_mhz"]), py(p["stall_pct"])
        lx_t, ly_t, anc, leader = label_specs[p["nom_mhz"]]
        if leader:
            e.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{lx_t:.1f}" y2="{ly_t+4:.1f}" stroke="#64748b" stroke-width="1.0" stroke-dasharray="2,2"/>')
        e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.3" fill="#ea580c" stroke="#ffffff" stroke-width="1.5"/>')
        lbl = f"{p['nom_mhz']} MHz: {fmt_vi(p['stall_pct'], 2)}% (TD: {fmt_vi(p['recon_pct'], 2)}%)"
        e.append(f'<text x="{lx_t:.1f}" y="{ly_t:.1f}" text-anchor="{anc}" font-size="10.4" font-weight="bold" fill="#1e293b">{xml_esc(lbl)}</text>')

    # Legend explicitly separating reconstruction from out-of-sample prediction
    lx, ly = ml + pw - 356, mt + 54
    e.append(f'<rect x="{lx}" y="{ly}" width="344" height="58" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    e.append(f'<line x1="{lx+12}" y1="{ly+20}" x2="{lx+42}" y2="{ly+20}" stroke="#1d4ed8" stroke-width="2.4"/>')
    e.append(f'<text x="{lx+50}" y="{ly+24}" font-size="10.4" fill="#0f172a">Tái dựng (TD) từ bộ đếm C_dma, C_tile cùng lần chạy</text>')
    e.append(f'<circle cx="{lx+27}" cy="{ly+42}" r="5" fill="#ea580c" stroke="#ffffff" stroke-width="1.2"/>')
    e.append(f'<text x="{lx+50}" y="{ly+46}" font-size="10.4" fill="#0f172a">Đo silicon thật trên EP4CE10 (512→1536→512, W4)</text>')

    # Title & subtitle
    e.append(
        f'<text x="{W/2:.1f}" y="23" text-anchor="middle" font-size="13.2" font-weight="bold" fill="#0f172a">'
        "Hình 3a. Tái dựng Chu kỳ Stall Ping-Pong DMA từ Bộ đếm Phần cứng (60–120 MHz, EP4CE10)</text>"
    )
    e.append(
        f'<text x="{W/2:.1f}" y="42" text-anchor="middle" font-size="10.5" fill="#475569">'
        "Phương pháp tái dựng chu kỳ cùng lần chạy (không phải dự báo ngoài mẫu; dự báo khóa trước xem Hình 3b)</text>"
    )
    e.append(
        f'<text x="{ml + pw/2:.1f}" y="{H-16}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e293b">'
        "Tần số xung SDRAM HY57V641620FTP-H (MHz, Engine cố định @ 80 MHz)</text>"
    )
    e.append(
        f'<text x="20" y="{mt + ph/2:.1f}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e293b" transform="rotate(-90 20 {mt + ph/2:.1f})">'
        "Tỉ lệ chu kỳ chờ DMA / Tổng chu kỳ (%)</text>"
    )

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig3b_oos_stall_prediction(oos: dict, out_path: Path) -> None:
    """Plot locked RTL-schedule out-of-sample predictions vs silicon measurements across 3 unseen FFN shapes."""
    # Exact f_crit @ 90 MHz from oos_stall_validation_results.json
    s_specs = [
        ("ffn_192_384_192", "192→384→192 (stage_0, L1 f_crit = 92,42 MHz)", "#dc2626", "none"),
        ("ffn_256_576_256", "256→576→256 (stage_1..5 FFN1, L1 f_crit = 90,56 MHz)", "#2563eb", "5,4"),
        ("ffn_256_960_256", "256→960→256 (stage_1..5 FFN3, L2 f_crit = 88,10 MHz)", "#059669", "2,3"),
    ]
    freqs = [60, 90, 100, 120]

    # Exact SDRAM packet layout matching run_oos_stall_validation.py
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

    W, H = 780, 464
    ml, mr, mt, mb = 74, 28, 62, 68
    pw, ph = W - ml - mr, H - mt - mb
    xmin, xmax = 55.0, 125.0
    ymin, ymax = 0.0, 42.0

    def px(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * pw

    def py(y: float) -> float:
        return mt + (1.0 - (y - ymin) / (ymax - ymin)) * ph

    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    for yv in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
        yy = py(yv)
        stroke = "#cbd5e1" if yv == 0 else "#e2e8f0"
        e.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"/>')
        e.append(f'<text x="{ml-10}" y="{yy+4:.1f}" text-anchor="end" font-size="11.2" fill="#334155">{yv}%</text>')

    for xv in [60, 70, 80, 90, 100, 110, 120]:
        xx = px(xv)
        e.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#e2e8f0" stroke-width="1"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt+ph+19}" text-anchor="middle" font-size="11.2" fill="#334155">{xv}</text>')

    e.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#475569" stroke-width="1.2"/>')

    for sname, _lbl, col, dash in s_specs:
        pts = [f"{px(55.0 + 0.5*k):.2f},{py(eval_curve_pct(sname, 55.0 + 0.5*k)):.2f}" for k in range(141)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        e.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.2"{dash_attr}/>')
        for mhz in freqs:
            r = oos["per_frequency"][f"sdram_{mhz}mhz"]
            st_m = r["shapes"][sname]["combined"]["meas_stall_pct"]
            cx, cy = px(r["measured_sdram_mhz"]), py(st_m)
            e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.6" fill="{col}" stroke="#ffffff" stroke-width="1.3"/>')

    # Annotate 90 MHz layer regime split using JSON values (wrapped to stay inside plot box)
    r90_192 = oos["per_frequency"]["sdram_90mhz"]["shapes"]["ffn_192_384_192"]
    r90_960 = oos["per_frequency"]["sdram_90mhz"]["shapes"]["ffn_256_960_256"]
    x90 = px(90.0)
    e.append(f'<line x1="{x90:.1f}" y1="{py(19.0):.1f}" x2="{x90:.1f}" y2="{py(0.0):.1f}" stroke="#94a3b8" stroke-width="1.1" stroke-dasharray="3,3"/>')
    e.append(
        f'<text x="{x90+8:.1f}" y="{py(17.5):.1f}" font-size="10.2" font-weight="bold" fill="#1e293b">'
        "Tách chế độ tại 90 MHz:</text>"
    )
    e.append(
        f'<text x="{x90+8:.1f}" y="{py(14.9):.1f}" font-size="10.0" font-weight="bold" fill="#991b1b">'
        f"192→384 L1: {fmt_vi(r90_192['l1']['meas_stall_pct'],2)}% đo / {fmt_vi(r90_192['l1']['pred_stall_pct'],2)}% dự báo (nghẽn)</text>"
    )
    e.append(
        f'<text x="{x90+8:.1f}" y="{py(12.3):.1f}" font-size="10.0" font-weight="bold" fill="#065f46">'
        f"256→960 L2: {fmt_vi(r90_960['l2']['meas_stall_pct'],2)}% đo / {fmt_vi(r90_960['l2']['pred_stall_pct'],2)}% dự báo (bão hòa)</text>"
    )

    lx, ly = ml + pw - 392, mt + 12
    e.append(f'<rect x="{lx}" y="{ly}" width="380" height="86" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    for idx, (_sn, lbl, col, dash) in enumerate(s_specs):
        yy = ly + 20 + idx * 24
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        e.append(f'<line x1="{lx+12}" y1="{yy}" x2="{lx+42}" y2="{yy}" stroke="{col}" stroke-width="2.2"{dash_attr}/>')
        e.append(f'<circle cx="{lx+27}" cy="{yy}" r="4" fill="{col}" stroke="#ffffff" stroke-width="1"/>')
        e.append(f'<text x="{lx+50}" y="{yy+4}" font-size="10.3" fill="#0f172a">{xml_esc(lbl)}</text>')

    e.append(
        f'<text x="{lx+190}" y="{ly+104}" text-anchor="middle" font-size="9.7" font-weight="bold" fill="#0f172a">'
        "Đo silicon vs Dự báo RTL (chu kỳ TB 0,0125%, max 0,0558% vs BIST 3,550%)</text>"
    )
    e.append(
        f'<text x="{W/2:.1f}" y="23" text-anchor="middle" font-size="13.2" font-weight="bold" fill="#0f172a">'
        "Hình 3b. Dự báo Ngoài mẫu Khóa Trước từ Lịch RTL trên 3 Hình dạng FFN Zipformer-30M</text>"
    )
    e.append(
        f'<text x="{W/2:.1f}" y="42" text-anchor="middle" font-size="10.2" fill="#475569">'
        "Khóa 0 tham số fit + 1 MiB BIST độc lập · Trọng số/kích hoạt nguyên tổng hợp cố định hạt giống (lịch RTL bất biến dữ liệu)</text>"
    )
    e.append(
        f'<text x="{ml + pw/2:.1f}" y="{H-18}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e293b">'
        "Tần số xung SDRAM HY57V641620FTP-H (MHz, Engine @ 80 MHz)</text>"
    )
    e.append(
        f'<text x="20" y="{mt + ph/2:.1f}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e293b" transform="rotate(-90 20 {mt + ph/2:.1f})">'
        "Tỉ lệ chu kỳ chờ DMA / Tổng chu kỳ (%)</text>"
    )

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def generate_fig4_wer_pareto_and_tones(
    paired: dict, abl_tone: dict, out_path: Path
) -> None:
    """Two-panel figure: (a) Delta-WER 19-speaker 95% CIs on VIVOS Test; (b) 6-tone error rates FP32 vs W4A8."""
    vmodels = paired["vivos_clean_all_models_utt_vs_speaker"]
    w4g64_blk = abl_tone["controlled_w4_vs_w3_g64_ablation"]

    # Collect rows cleanly from JSON
    rows = [
        ("Bit-Exact W8A8", vmodels["30M_fpga_bitexact_w8a8"]["speaker_clustered_bootstrap_vs_fp32"], "#16a34a", "Đạt biên"),
        ("FFN W4 (g64, ĐC)", w4g64_blk["w4_g64_vs_fp32_spk"], "#d97706", "Chưa kết luận"),
        ("W8A8 ONNX (enc)", vmodels["30M_W8A8"]["speaker_clustered_bootstrap_vs_fp32"], "#d97706", "Chưa kết luận"),
        ("Bit-Exact W4A8", vmodels["30M_fpga_bitexact_w4a8"]["speaker_clustered_bootstrap_vs_fp32"], "#ea580c", "Chưa kết luận"),
        ("FFN W3 (g64, ĐC)", w4g64_blk["w3_g64_vs_fp32_spk"], "#c2410c", "Kém FP32"),
        ("All W4 (per-ch)", vmodels["30M_all_w4_perch"]["speaker_clustered_bootstrap_vs_fp32"], "#c2410c", "Kém FP32"),
        ("FFN W2 (u4, g64)", vmodels["30M_ffn_w2_u4_g64"]["speaker_clustered_bootstrap_vs_fp32"], "#b91c1c", "Kém rõ rệt"),
    ]

    W, H = 960, 454
    e: list[str] = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    e.append(
        f'<text x="{W/2:.1f}" y="23" text-anchor="middle" font-size="13.2" font-weight="bold" fill="#0f172a">'
        "Hình 4. Đánh đổi Độ chính xác Lượng tử hóa và Phân bố Lỗi 6 Thanh điệu trên VIVOS Test (760 câu, 19 người nói)</text>"
    )

    # ---------------- Panel (a): Forest Plot Delta-WER ----------------
    ml1, mt1, pw1, ph1 = 152, 68, 336, 294
    xmin1, xmax1 = -0.30, 2.35

    def px1(v: float) -> float:
        return ml1 + (v - xmin1) / (xmax1 - xmin1) * pw1

    e.append(
        f'<text x="{ml1 + pw1/2:.1f}" y="50" text-anchor="middle" font-size="11.2" font-weight="bold" fill="#1e293b">'
        "(a) ΔWER so với FP32 (4,610%) &amp; CI 95% theo 19 người nói</text>"
    )

    # Shaded NI zone <= +0.10
    e.append(f'<rect x="{ml1:.1f}" y="{mt1:.1f}" width="{px1(0.10) - ml1:.1f}" height="{ph1:.1f}" fill="#f0fdf4" opacity="0.8"/>')

    for xv in [0.0, 0.5, 1.0, 1.5, 2.0]:
        xx = px1(xv)
        col = "#94a3b8" if xv == 0.0 else "#e2e8f0"
        e.append(f'<line x1="{xx:.1f}" y1="{mt1}" x2="{xx:.1f}" y2="{mt1+ph1}" stroke="{col}" stroke-width="1"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt1+ph1+18}" text-anchor="middle" font-size="10.2" fill="#334155">{fmt_vi(xv, 1)}%</text>')

    # NI threshold +0.10%
    x_ni = px1(0.10)
    e.append(f'<line x1="{x_ni:.1f}" y1="{mt1}" x2="{x_ni:.1f}" y2="{mt1+ph1}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,3"/>')
    e.append(f'<text x="{x_ni+4:.1f}" y="{mt1+14}" font-size="9.2" font-weight="bold" fill="#b91c1c">Biên +0,10%</text>')

    e.append(f'<rect x="{ml1}" y="{mt1}" width="{pw1}" height="{ph1}" fill="none" stroke="#475569" stroke-width="1.2"/>')

    row_step = ph1 / len(rows)
    for idx, (label, blk, col, status) in enumerate(rows):
        yy = mt1 + (idx + 0.5) * row_step
        if idx > 0:
            ysep = mt1 + idx * row_step
            e.append(f'<line x1="{ml1}" y1="{ysep:.1f}" x2="{ml1+pw1}" y2="{ysep:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        d = float(blk["delta_wer_pct"])
        ci_lo, ci_hi = float(blk["delta_wer_ci95"][0]), float(blk["delta_wer_ci95"][1])
        x_pt, x_lo, x_hi = px1(d), px1(ci_lo), px1(ci_hi)

        e.append(f'<text x="{ml1-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.2" font-weight="bold" fill="#1e293b">{xml_esc(label)}</text>')
        e.append(f'<line x1="{x_lo:.1f}" y1="{yy:.1f}" x2="{x_hi:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="2.2"/>')
        e.append(f'<line x1="{x_lo:.1f}" y1="{yy-5:.1f}" x2="{x_lo:.1f}" y2="{yy+5:.1f}" stroke="{col}" stroke-width="1.8"/>')
        e.append(f'<line x1="{x_hi:.1f}" y1="{yy-5:.1f}" x2="{x_hi:.1f}" y2="{yy+5:.1f}" stroke="{col}" stroke-width="1.8"/>')
        e.append(f'<circle cx="{x_pt:.1f}" cy="{yy:.1f}" r="4.4" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>')

        txt = f"{fmt_vi(d, 3, signed=True)} [{fmt_vi(ci_lo, 3, signed=True)}; {fmt_vi(ci_hi, 3, signed=True)}]"
        if idx < 6:
            e.append(f'<text x="{x_hi + 6:.1f}" y="{yy+3.5:.1f}" font-size="9.0" fill="#1e293b">{txt}</text>')
        else:
            e.append(f'<text x="{x_pt:.1f}" y="{yy-10.0:.1f}" text-anchor="middle" font-size="9.0" font-weight="bold" fill="#991b1b">{txt}</text>')

    # Controlled W4_g64 vs W3_g64 callout in empty bottom-left pocket (left of FFN W2 bar)
    d34 = w4g64_blk["w3_g64_minus_w4_g64_spk"]
    e.append(f'<rect x="{ml1+6}" y="{mt1+248}" width="166" height="40" rx="4" fill="#fffbeb" stroke="#f59e0b" stroke-width="1"/>')
    e.append(f'<text x="{ml1+13}" y="{mt1+264}" font-size="8.7" font-weight="bold" fill="#92400e">Đối chứng g64 (4 → 3 bit):</text>')
    e.append(
        f'<text x="{ml1+13}" y="{mt1+280}" font-size="8.7" fill="#78350f">'
        f"W3 − W4 = {fmt_vi(d34['delta_wer_pct'],3,signed=True)}% [{fmt_vi(d34['delta_wer_ci95'][0],3,signed=True)}; {fmt_vi(d34['delta_wer_ci95'][1],3,signed=True)}]</text>"
    )
    e.append(
        f'<text x="{ml1 + pw1/2:.1f}" y="{mt1+ph1+38}" text-anchor="middle" font-size="10.4" font-weight="bold" fill="#1e293b">'
        "Độ lệch ΔWER so với chuẩn FP32 (điểm %)</text>"
    )

    # ---------------- Panel (b): 6-Tone Error Rates ----------------
    ml2, mt2, pw2, ph2 = 556, 68, 376, 294
    ymax2 = 1.75

    def py2(v: float) -> float:
        return mt2 + (1.0 - v / ymax2) * ph2

    e.append(
        f'<text x="{ml2 + pw2/2:.1f}" y="50" text-anchor="middle" font-size="11.2" font-weight="bold" fill="#1e293b">'
        "(b) Tỉ lệ lỗi theo 6 thanh trên âm tiết cùng gốc (~96% phủ)</text>"
    )
    for yv in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]:
        yy = py2(yv)
        stroke = "#cbd5e1" if yv == 0.0 else "#e2e8f0"
        e.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"/>')
        e.append(f'<text x="{ml2-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.2" fill="#334155">{fmt_vi(yv, 1)}%</text>')
    e.append(f'<rect x="{ml2}" y="{mt2}" width="{pw2}" height="{ph2}" fill="none" stroke="#475569" stroke-width="1.2"/>')

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

    bw2 = 20.0
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
        e.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{bw2}" height="{py2(0)-y1:.1f}" fill="#3b82f6" stroke="#1d4ed8" stroke-width="1"/>')
        e.append(f'<rect x="{x2:.1f}" y="{y2:.1f}" width="{bw2}" height="{py2(0)-y2:.1f}" fill="#f97316" stroke="#c2410c" stroke-width="1"/>')
        # Exact counts right above each bar
        e.append(f'<text x="{x1+bw2/2:.1f}" y="{y1-4:.1f}" text-anchor="middle" font-size="8.8" fill="#1e3a8a">{c_fp}</text>')
        e.append(f'<text x="{x2+bw2/2:.1f}" y="{y2-4:.1f}" text-anchor="middle" font-size="8.8" font-weight="bold" fill="#9a3412">{c_w4}</text>')
        e.append(f'<text x="{cx:.1f}" y="{mt2+ph2+18}" text-anchor="middle" font-size="10.4" font-weight="bold" fill="#1e293b">{tname}</text>')

    # Legend & summary note in Panel (b)
    fp_all = abl_tone["tone_6x6_matrices"]["30M_FP32"]
    w4_all = abl_tone["tone_6x6_matrices"]["30M_fpga_bitexact_w4a8"]
    td_all = abl_tone["tone_6x6_speaker_bootstrap_deltas"]["w4a8_bitexact_minus_fp32"]
    nga_ci_hi = td_all["per_tone_delta"]["nga"]["ci95"][1]
    e.append(f'<rect x="{ml2+8}" y="{mt2+8}" width="218" height="66" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    e.append(f'<rect x="{ml2+16}" y="{mt2+17}" width="12" height="10" fill="#3b82f6" stroke="#1d4ed8" stroke-width="0.8"/>')
    e.append(f'<text x="{ml2+34}" y="{mt2+26}" font-size="9.3" fill="#0f172a">FP32: {fp_all["overall_same_base_tone_errors"]}/{fp_all["aligned_same_base_syllables"]:,} ({fmt_vi(fp_all["overall_same_base_tone_err_pct"],3)}%)</text>'.replace(",", "."))
    # Wait! Be careful not to replace decimal comma! Format denominator separately:
    e[-1] = (
        f'<text x="{ml2+34}" y="{mt2+26}" font-size="9.3" fill="#0f172a">'
        f"FP32: {fp_all['overall_same_base_tone_errors']}/7.416 ({fmt_vi(fp_all['overall_same_base_tone_err_pct'],3)}%)</text>"
    )
    e.append(f'<rect x="{ml2+16}" y="{mt2+35}" width="12" height="10" fill="#f97316" stroke="#c2410c" stroke-width="0.8"/>')
    e.append(
        f'<text x="{ml2+34}" y="{mt2+44}" font-size="9.3" fill="#0f172a">'
        f"W4A8: {w4_all['overall_same_base_tone_errors']}/7.412 ({fmt_vi(w4_all['overall_same_base_tone_err_pct'],3)}%)</text>"
    )
    e.append(
        f'<text x="{ml2+16}" y="{mt2+62}" font-size="8.8" font-weight="bold" fill="#475569">'
        f"Δ chung: {fmt_vi(td_all['overall_delta_err_pct'],3,signed=True)}% · Ngã CI trên: {fmt_vi(nga_ci_hi,3,signed=True)}%</text>"
    )

    e.append(f'<text x="{ml2 + pw2/2:.1f}" y="{mt2+ph2+38}" text-anchor="middle" font-size="10.4" font-weight="bold" fill="#1e293b">Thanh điệu tiếng Việt (nhãn trên cột: số âm tiết sai thanh)</text>')
    e.append(f'<text x="514" y="{mt2+ph2/2:.1f}" text-anchor="middle" font-size="10.8" font-weight="bold" fill="#1e293b" transform="rotate(-90 514 {mt2+ph2/2:.1f})">Tỉ lệ nhầm thanh / cùng gốc (%)</text>')

    e.append(
        f'<text x="{W/2:.1f}" y="{H-14}" text-anchor="middle" font-size="10.0" fill="#475569">'
        "Ghi chú: (a) ĐC = đối chứng cùng group-64; phần lớn CI đều giao 0 hoặc vượt biên +0,10%; (b) Ngã có cỡ mẫu nhỏ (357 âm tiết) nên CI 19 người nói rộng tới +1,260%.</text>"
    )

    out_path.write_text(wrap_svg(W, H, e), encoding="utf-8")


def build_padded_t500_mcu_data(g5b: dict, img: dict) -> dict:
    """Recompute T=500 FFN-only break-even under full 4-frame tail-padded tiles (984 block-tiles).

    Matches closed-loop hardware behavior: every tail tile (1..3 valid frames) is padded to
    T_TILE = 4 frames on both bidirectional activation transfers and FPGA execution cycles.
    """
    t500_proj = g5b["part_a_full_sequence_measurement"]["t500_normalized_projection"]
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
    sdram_bytes_read = 0
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
        sdram_bytes_read += n_t * (
            int(blk["l1_meta"]["total_bytes"]) + int(blk["l2_meta"]["total_bytes"])
        )

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

    if ffn_macs != int(t500_proj["ffn_macs_total"]):
        raise ValueError(f"FFN MAC mismatch: {ffn_macs} != {t500_proj['ffn_macs_total']}")
    if total_block_tiles != int(t500_proj["total_tiles_n"]):
        raise ValueError(f"Tile mismatch: {total_block_tiles} != {t500_proj['total_tiles_n']}")
    if sdram_bytes_read != int(t500_proj["sdram_bytes_read"]):
        raise ValueError(f"SDRAM read mismatch: {sdram_bytes_read} != {t500_proj['sdram_bytes_read']}")

    ffn_gmac = ffn_macs / 1e9
    t_fpga_s = pred_cycles_total / 80e6
    act_mb = act_io_bytes_padded / 1e6
    act_mib = act_io_bytes_padded / float(1024 * 1024)
    g_max = ffn_gmac / t_fpga_s
    knee = act_mb / t_fpga_s

    t_jtag_io_s = (act_io_bytes_padded / 2.0) / wr_bps + (act_io_bytes_padded / 2.0) / rd_bps
    bw_jtag_eff = act_mb / t_jtag_io_s

    bw_grid = [bw_jtag_eff, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    sweep_points = []
    for bw in bw_grid:
        t_io = act_mb / bw
        g_seq = ffn_gmac / (t_fpga_s + t_io)
        g_ovl = ffn_gmac / max(t_fpga_s, t_io)
        io_share = 100.0 * t_io / (t_fpga_s + t_io)
        sweep_points.append({
            "bus_bw_MBps": round(bw, 6 if bw < 0.02 else 2),
            "t_io_s": round(t_io, 4),
            "breakeven_seq_gmac_s": round(g_seq, 4),
            "breakeven_ovl_gmac_s": round(g_ovl, 4),
            "io_share_seq_pct": round(io_share, 2),
        })

    return {
        "model": "Zipformer-30M FFN-only (36 blocks, 72 Linear layers)",
        "reference_audio": {"t_fbank": 500, "duration_s": 5.0},
        "projection_convention": (
            "Full 4-frame tail-padded 984 block-tiles matching closed-loop hardware execution "
            "(predict_layer_rtl @ 80/100 MHz, bidirectional JTAG from T=263 write/read rates)"
        ),
        "total_block_tiles_padded": total_block_tiles,
        "pred_total_cycles": pred_cycles_total,
        "ffn_gmac": round(ffn_gmac, 9),
        "t_fpga_s": round(t_fpga_s, 6),
        "ffn_rtf_vs_5s": round(t_fpga_s / 5.0, 5),
        "act_io_bytes": act_io_bytes_padded,
        "act_io_MB_decimal": round(act_mb, 6),
        "act_io_MiB_binary": round(act_mib, 4),
        "jtag_eff_bw_MBps": round(bw_jtag_eff, 6),
        "jtag_io_time_s": round(t_jtag_io_s, 2),
        "jtag_io_rtf_vs_5s": round(t_jtag_io_s / 5.0, 2),
        "fpga_asymptote_gmac_s": round(g_max, 4),
        "overlap_knee_bw_MBps": round(knee, 4),
        "sweep_points": sweep_points,
    }


def generate_fig5_mcu_breakeven(data: dict, g5b: dict, out_path: Path) -> None:
    ffn_gmac = data["ffn_gmac"]
    t_fpga_s = data["t_fpga_s"]
    act_bytes = float(data["act_io_bytes"])
    act_mb = data["act_io_MB_decimal"]
    g_max = data["fpga_asymptote_gmac_s"]

    # Effective bidirectional activation JTAG bandwidth from measured write/read rates
    jtag_rates = g5b["part_a_full_sequence_measurement"]["measured_full_sequence_t263"]
    wr_bps = float(jtag_rates["measured_jtag_write_Bps"])
    rd_bps = float(jtag_rates["measured_jtag_read_Bps"])
    t_io_jtag = (act_bytes / 2.0) / wr_bps + (act_bytes / 2.0) / rd_bps
    bw_jtag_eff = act_mb / t_io_jtag  # ~0.0104 MB/s decimal

    W, H = 780, 464
    ml, mr, mt, mb = 74, 28, 62, 68
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
    e.append(f'<rect x="{ml}" y="{y_m7_top:.1f}" width="{pw}" height="{y_m7_bot - y_m7_top:.1f}" fill="#fef9c3" opacity="0.82"/>')

    n_pts = 200
    bws = [10 ** (lx_min + (lx_max - lx_min) * i / (n_pts - 1)) for i in range(n_pts)]
    seq_poly = [f"{px(b):.2f},{py(g_seq(b)):.2f}" for b in bws]
    ovl_pts = [f"{px(b):.2f},{py(g_ovl(b)):.2f}" for b in bws]
    poly_all = seq_poly + [f"{px(bws[-1]):.2f},{py(0.0):.2f}", f"{px(bws[0]):.2f},{py(0.0):.2f}"]
    e.append(f'<polygon points="{" ".join(poly_all)}" fill="#dbeafe" opacity="0.55"/>')

    for yv in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]:
        yy = py(yv)
        stroke = "#cbd5e1" if yv == 0.0 else "#e2e8f0"
        e.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"/>')
        e.append(f'<text x="{ml-10}" y="{yy+4:.1f}" text-anchor="end" font-size="11.2" fill="#334155">{fmt_vi(yv, 1)}</text>')

    xticks = [(0.01, "0,01"), (0.03, "0,03"), (0.1, "0,1"), (0.3, "0,3"), (1.0, "1"), (2.0, "2"), (5.0, "5"), (10.0, "10"), (30.0, "30")]
    for xv, xlbl in xticks:
        xx = px(xv)
        e.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#e2e8f0" stroke-width="1"/>')
        e.append(f'<text x="{xx:.1f}" y="{mt+ph+19}" text-anchor="middle" font-size="11.2" fill="#334155">{xlbl}</text>')

    e.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#475569" stroke-width="1.2"/>')

    y_ceil = py(g_max)
    e.append(f'<line x1="{ml}" y1="{y_ceil:.1f}" x2="{ml+pw}" y2="{y_ceil:.1f}" stroke="#0f766e" stroke-width="1.4" stroke-dasharray="6,4"/>')
    e.append(f'<text x="{ml+10}" y="{y_ceil-6:.1f}" font-size="10.6" font-weight="bold" fill="#0f766e">Trần FPGA tối đa khi bus → ∞: {fmt_vi(g_max, 2)} GMAC/s (t_FPGA = {fmt_vi(t_fpga_s, 3)} s)</text>')

    e.append(f'<text x="{ml+pw-8}" y="{mt+20}" text-anchor="end" font-size="10.2" font-weight="bold" fill="#475569">Cortex-A9 NEON (~1,4–3,9 GMAC/s)</text>')
    e.append(f'<text x="{ml+pw-8}" y="{mt+35}" text-anchor="end" font-size="9.5" fill="#64748b">(Nhanh hơn trần EP4CE10 → chạy trên host)</text>')

    e.append(f'<text x="{ml+10}" y="{py(0.39):.1f}" font-size="10.4" font-weight="bold" fill="#854d0e">Cortex-M7 ước lượng (~0,3–0,5 GMAC/s, DSP SIMD)</text>')
    e.append(f'<text x="{px(4.8):.1f}" y="{py(0.20):.1f}" text-anchor="middle" font-size="10.6" font-weight="bold" fill="#1e40af">Miền offload lý tưởng cho host</text>')
    e.append(f'<text x="{px(4.8):.1f}" y="{py(0.08):.1f}" text-anchor="middle" font-size="9.6" fill="#1e3a8a">(trước khi tính chi phí điều khiển thực)</text>')

    e.append(f'<polyline points="{" ".join(ovl_pts)}" fill="none" stroke="#9333ea" stroke-width="2.0" stroke-dasharray="5,4"/>')
    e.append(f'<polyline points="{" ".join(seq_poly)}" fill="none" stroke="#1d4ed8" stroke-width="2.6"/>')

    g_jtag_eff = g_seq(bw_jtag_eff)
    markers = [
        (bw_jtag_eff, f"JTAG 2 chiều ({fmt_vi(bw_jtag_eff, 4)} MB/s): {fmt_vi(g_jtag_eff, 4)}", (8, -10, "start")),
        (1.0, f"1 MB/s: {fmt_vi(g_seq(1.0), 3)}", (-10, -10, "end")),
        (5.0, f"5 MB/s: {fmt_vi(g_seq(5.0), 3)}", (-8, 18, "end")),
        (10.0, f"10 MB/s: {fmt_vi(g_seq(10.0), 3)}", (8, 16, "start")),
    ]
    for bw, lbl, (dx, dy, anc) in markers:
        cx, cy = px(bw), py(g_seq(bw))
        e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.8" fill="#dc2626" stroke="#ffffff" stroke-width="1.4"/>')
        e.append(f'<text x="{cx+dx:.1f}" y="{cy+dy:.1f}" text-anchor="{anc}" font-size="10.4" font-weight="bold" fill="#1e293b">{xml_esc(lbl)}</text>')

    lx, ly = ml + 10, mt + 10
    e.append(f'<rect x="{lx}" y="{ly}" width="332" height="58" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
    e.append(f'<line x1="{lx+12}" y1="{ly+20}" x2="{lx+42}" y2="{ly+20}" stroke="#1d4ed8" stroke-width="2.6"/>')
    e.append(f'<text x="{lx+50}" y="{ly+24}" font-size="10.0" fill="#0f172a">Hòa vốn tuần tự: G = 1,236 / (t_FPGA + t_IO)</text>')
    e.append(f'<line x1="{lx+12}" y1="{ly+42}" x2="{lx+42}" y2="{ly+42}" stroke="#9333ea" stroke-width="2.0" stroke-dasharray="5,4"/>')
    e.append(f'<text x="{lx+50}" y="{ly+46}" font-size="10.0" fill="#0f172a">Cận trên gối đầu I/O: max(t_FPGA, t_IO)</text>')

    e.append(
        f'<text x="{W/2:.1f}" y="23" text-anchor="middle" font-size="13.0" font-weight="bold" fill="#0f172a">'
        "Hình 5. Đường cong Hòa vốn Thông lượng FFN-Only cho Host Nhúng theo Băng thông Bus (T = 500)</text>"
    )
    e.append(
        f'<text x="{W/2:.1f}" y="42" text-anchor="middle" font-size="10.2" fill="#475569">'
        f"Dự phóng {data['total_block_tiles_padded']} tile đệm đuôi ({fmt_vi(act_mb, 2)} MB I/O) trước khi tính chi phí điều khiển (thực đo T=491 thêm 38,32 s)</text>"
    )
    e.append(
        f'<text x="{ml + pw/2:.1f}" y="{H-18}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e293b">'
        "Băng thông liên kết Host↔FPGA (MB/s thập phân, thang log)</text>"
    )
    e.append(
        f'<text x="20" y="{mt + ph/2:.1f}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e293b" transform="rotate(-90 20 {mt + ph/2:.1f})">'
        "Thông lượng FFN hòa vốn của Host (GMAC/s)</text>"
    )

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
    (BUILD_DIR / "mcu_breakeven_analysis.json").write_text(
        json.dumps(mcu_data, indent=2) + "\n", encoding="utf-8"
    )

    outputs = [
        FIG_DIR / "fig1_hybrid_architecture.svg",
        FIG_DIR / "fig2_m9k_bandwidth_tradeoff.svg",
        FIG_DIR / "fig3a_sdram_stall_reconstruction.svg",
        FIG_DIR / "fig3b_oos_stall_prediction.svg",
        FIG_DIR / "fig4_wer_pareto_and_tones.svg",
        FIG_DIR / "fig5_mcu_breakeven.svg",
    ]

    generate_fig1_hybrid_architecture(outputs[0])
    generate_fig2_m9k_bandwidth_tradeoff(mem_data, g4b, outputs[1])
    generate_fig3a_sdram_stall_reconstruction(g4b, outputs[2])
    generate_fig3b_oos_stall_prediction(oos, outputs[3])
    generate_fig4_wer_pareto_and_tones(paired, abl_tone, outputs[4])
    generate_fig5_mcu_breakeven(mcu_data, g5b_data, outputs[5])

    for p in outputs:
        ET.parse(p)
        print(f"Verified SVG XML: {p} ({p.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
