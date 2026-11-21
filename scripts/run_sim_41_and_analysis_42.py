#!/usr/bin/env python3
"""Section 4.1 (RTL Simulation with real measured clocks + initial phase sweep + 60/60 drift check)
and Section 4.2 (Offline statistical & residual analysis on existing 78-pt & 0..31 datasets).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_deep_rtl_validation_part2 import SHAPES_6, predict_layer_param
from run_oos_stall_validation import make_synthetic_integer_ffn

ROOT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT_DIR / "build"
OUT_JSON = BUILD_DIR / "analysis_41_42_results.json"


def run_41_sims() -> dict:
    # Compile tb_offset_sim parameterized or use tb_78pt_sim-style plusargs?
    # Let's compile a tiny wrapper or use tb_offset_sim with real clock + phase sweep via python-generated or plusarg tb.
    vvp_90 = BUILD_DIR / "tb_41_offset90.vvp"
    tb_code = r"""`timescale 1ns/1ps
module tb_41_offset90;
    reg clk_eng = 1'b0, clk_sdram = 1'b0, rst_n = 1'b0;
    real eng_half = 6.253439;   // 79.956 MHz
    real sd_half  = 5.555926;   // 89.994 MHz
    real init_sd_delay = 0.0;
    integer ref_delay = 0;
    initial begin
        if (!$value$plusargs("ENG_HALF=%f", eng_half)) eng_half = 6.253439;
        if (!$value$plusargs("SD_HALF=%f", sd_half)) sd_half = 5.555926;
        if (!$value$plusargs("INIT_SD_DELAY=%f", init_sd_delay)) init_sd_delay = 0.0;
        if (!$value$plusargs("REF_DELAY=%d", ref_delay)) ref_delay = 0;
        #(init_sd_delay);
        forever #(sd_half) clk_sdram = ~clk_sdram;
    end
    always #(eng_half) clk_eng = ~clk_eng;

    wire sdram_cke, sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n;
    wire [1:0] sdram_ba, sdram_dqm;
    wire [11:0] sdram_addr;
    wire [15:0] sdram_dq_out;
    wire sdram_dq_oe;
    reg [15:0] sdram_dq_drv;
    reg sdram_dq_en;
    wire [15:0] sdram_dq = sdram_dq_oe ? sdram_dq_out : (sdram_dq_en ? sdram_dq_drv : 16'h0000);
    reg [11:0] sdram_act_row [0:3];
    reg sdram_rd_v0, sdram_rd_v1;
    reg [15:0] sdram_rd_d0, sdram_rd_d1;
    always @(posedge clk_sdram) begin
        if (!rst_n) begin sdram_dq_en <= 0; sdram_rd_v0 <= 0; sdram_rd_v1 <= 0; end
        else begin
            sdram_dq_en <= 0; sdram_rd_v1 <= sdram_rd_v0; sdram_rd_d1 <= sdram_rd_d0;
            if (sdram_rd_v1) begin sdram_dq_en <= 1; sdram_dq_drv <= sdram_rd_d1; end
            sdram_rd_v0 <= 0;
            if (sdram_cke && !sdram_cs_n) case ({sdram_ras_n, sdram_cas_n, sdram_we_n})
                3'b011: sdram_act_row[sdram_ba] <= sdram_addr;
                3'b101: begin sdram_rd_v0 <= 1; sdram_rd_d0 <= {sdram_ba, sdram_act_row[sdram_ba][5:0], sdram_addr[7:0]}; end
                default: ;
            endcase
        end
    end

    wire sdram_req_tog, sdram_ack_tog, sdram_ready, sdram_dma_w128_valid;
    wire [1:0] sdram_req_op; wire [18:0] sdram_req_addr; wire [19:0] sdram_req_len;
    wire [127:0] sdram_req_wdata, sdram_rdata, sdram_dma_w128_data;
    wire [2:0] sdram_cfg_rd_lat; wire [8:0] sdram_dma_w128_idx;
    wire [31:0] sdram_bist_wr_cyc, sdram_bist_rd_cyc, sdram_bist_err; wire [63:0] sdram_cyc_cnt;

    sdram_ctrl #(.CLK_MHZ(90), .INIT_WAIT_CYC(40)) u_sdram (
        .clk_sdram(clk_sdram), .clk_eng(clk_eng), .rst_n(rst_n),
        .eng_req_tog(sdram_req_tog), .eng_req_op(sdram_req_op), .eng_addr(sdram_req_addr),
        .eng_len(sdram_req_len), .eng_wdata(sdram_req_wdata), .eng_rd_lat(sdram_cfg_rd_lat),
        .eng_ack_tog(sdram_ack_tog), .eng_rdata(sdram_rdata),
        .eng_bist_wr_cyc(sdram_bist_wr_cyc), .eng_bist_rd_cyc(sdram_bist_rd_cyc), .eng_bist_err(sdram_bist_err),
        .eng_sdram_ready(sdram_ready), .eng_dma_w128_valid(sdram_dma_w128_valid),
        .eng_dma_w128_data(sdram_dma_w128_data), .eng_dma_w128_idx(sdram_dma_w128_idx),
        .eng_sdram_cyc_cnt(sdram_cyc_cnt),
        .sdram_cke(sdram_cke), .sdram_cs_n(sdram_cs_n), .sdram_ras_n(sdram_ras_n),
        .sdram_cas_n(sdram_cas_n), .sdram_we_n(sdram_we_n), .sdram_ba(sdram_ba),
        .sdram_addr(sdram_addr), .sdram_dqm(sdram_dqm), .sdram_dq_in(sdram_dq),
        .sdram_dq_out(sdram_dq_out), .sdram_dq_oe(sdram_dq_oe)
    );
    reg cmd_valid = 0; reg [63:0] cmd_word = 0; wire [63:0] resp_word; wire status_busy, status_done;
    zipformer_engine #(.NUM_LANES(16), .MAX_T_TILE(4)) u_engine (
        .clk(clk_eng), .rst_n(rst_n), .cmd_valid(cmd_valid), .cmd_word(cmd_word),
        .resp_word(resp_word), .status_busy(status_busy), .status_done(status_done),
        .sdram_req_tog(sdram_req_tog), .sdram_req_op(sdram_req_op), .sdram_req_addr(sdram_req_addr),
        .sdram_req_len(sdram_req_len), .sdram_req_wdata(sdram_req_wdata), .sdram_cfg_rd_lat(sdram_cfg_rd_lat),
        .sdram_ack_tog(sdram_ack_tog), .sdram_rdata(sdram_rdata),
        .sdram_bist_wr_cyc(sdram_bist_wr_cyc), .sdram_bist_rd_cyc(sdram_bist_rd_cyc), .sdram_bist_err(sdram_bist_err),
        .sdram_ready(sdram_ready), .sdram_dma_w128_valid(sdram_dma_w128_valid),
        .sdram_dma_w128_data(sdram_dma_w128_data), .sdram_dma_w128_idx(sdram_dma_w128_idx),
        .sdram_cyc_cnt(sdram_cyc_cnt)
    );
    task send_cmd(input [7:0] op, input [23:0] addr, input [31:0] wdata);
        begin @(negedge clk_eng); cmd_valid = 1; cmd_word = {op, addr, wdata}; @(posedge clk_eng); @(negedge clk_eng); cmd_valid = 0; end
    endtask
    task read_reg(input [15:0] sel, output [63:0] val);
        begin send_cmd(8'h06, {8'd0, sel}, 32'd0); @(posedge clk_eng); val = resp_word; end
    endtask
    task run_pair(input [18:0] b1, input [18:0] b2, output integer l1_t, output integer l2_t);
        reg [63:0] r1;
        begin
            send_cmd(8'h07, 24'd0, {1'b0, 2'd0, 4'd1, 8'd192, 8'd4, 2'd1, 1'b1, 2'd0});
            send_cmd(8'h07, 24'd1, {8'd48, 8'd0, 8'd0, 8'd16});
            send_cmd(8'h07, 24'd2, {8'd48, 24'd0});
            send_cmd(8'h07, 24'd3, {2'd0, 1'b1, 8'd12, 1'b0, 8'd40, 1'b0, 1'b0, 2'd0, 8'd0});
            send_cmd(8'h07, 24'd4, {7'd0, 9'd96, 3'd4, 4'd1, 8'd40, 1'b1});
            send_cmd(8'h07, 24'd5, {13'd0, b1});
            send_cmd(8'h08, 24'd0, 32'd0);
            @(posedge clk_eng); while (!status_done) @(posedge clk_eng);
            read_reg(16'h2001, r1); l1_t = r1[31:0];

            send_cmd(8'h07, 24'd0, {1'b0, 2'd0, 4'd1, 8'd64, 8'd4, 2'd0, 1'b1, 2'd0});
            send_cmd(8'h07, 24'd1, {8'd0, 8'd0, 8'd48, 8'd16});
            send_cmd(8'h07, 24'd2, {8'd0, 24'd0});
            send_cmd(8'h07, 24'd3, {2'd0, 1'b0, 8'd40, 1'b0, 8'd12, 1'b0, 1'b0, 2'd1, 8'd0});
            send_cmd(8'h07, 24'd4, {7'd0, 9'd160, 3'd4, 4'd2, 8'd12, 1'b1});
            send_cmd(8'h07, 24'd5, {13'd0, b2});
            send_cmd(8'h08, 24'd0, 32'd0);
            @(posedge clk_eng); while (!status_done) @(posedge clk_eng);
            read_reg(16'h2001, r1); l2_t = r1[31:0];
        end
    endtask
    integer l1_1, l2_1, l1_15, l2_15;
    initial begin
        rst_n = 0; #200; rst_n = 1;
        while (!sdram_ready) @(posedge clk_eng);
        repeat (20 + ref_delay) @(posedge clk_eng);
        run_pair(19'd16385, 19'd32769, l1_1, l2_1);
        run_pair(19'd16399, 19'd32783, l1_15, l2_15);
        $display("RES l1_1=%0d l2_1=%0d l1_15=%0d l2_15=%0d dl1=%0d dl2=%0d dtot=%0d",
                 l1_1, l2_1, l1_15, l2_15, l1_1 - l1_15, l2_1 - l2_15, (l1_1+l2_1)-(l1_15+l2_15));
        $finish;
    end
endmodule
"""
    tb_path = ROOT_DIR / "tb" / "tb_41_offset90.v"
    tb_path.write_text(tb_code, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-DSIMULATION", "-o", str(vvp_90),
         "rtl/sdram_ctrl.v", "rtl/zipformer_engine.v", str(tb_path)],
        cwd=str(ROOT_DIR), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # 11 refresh phases x 6 initial clock phase offsets (0..10 ns step 2 ns) at measured 79.956 / 89.994 MHz
    eng_half = 500.0 / 79.956
    sd_half = 500.0 / 89.994
    dl1s, dl2s, dtots, o1_tots, o15_tots = [], [], [], [], []
    for ref_d in range(0, 1001, 100):
        for init_d in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
            out = subprocess.check_output(
                ["vvp", str(vvp_90), f"+ENG_HALF={eng_half:.6f}", f"+SD_HALF={sd_half:.6f}",
                 f"+INIT_SD_DELAY={init_d:.3f}", f"+REF_DELAY={ref_d}"],
                cwd=str(ROOT_DIR), text=True
            )
            m = re.search(r"RES l1_1=(\d+) l2_1=(\d+) l1_15=(\d+) l2_15=(\d+) dl1=(-?\d+) dl2=(-?\d+) dtot=(-?\d+)", out)
            assert m
            l1_1, l2_1, l1_15, l2_15 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            dl1s.append(int(m.group(5)))
            dl2s.append(int(m.group(6)))
            dtots.append(int(m.group(7)))
            o1_tots.append(l1_1 + l2_1)
            o15_tots.append(l1_15 + l2_15)

    # 60/60 MHz check from three_way_78pt_sim_decomp.json
    tw = json.loads((BUILD_DIR / "three_way_78pt_sim_decomp.json").read_text())
    e60_s60 = [r for r in tw["rows"] if r["tag"] == "e60_s60"]

    return {
        "offset_realclk_66runs": {
            "n_runs": len(dtots),
            "o1_mean": round(float(np.mean(o1_tots)), 2),
            "o15_mean": round(float(np.mean(o15_tots)), 2),
            "dl1_mean": round(float(np.mean(dl1s)), 2),
            "dl1_sd": round(float(np.std(dl1s, ddof=1)), 2),
            "dl1_range": [min(dl1s), max(dl1s)],
            "dl2_mean": round(float(np.mean(dl2s)), 2),
            "dl2_sd": round(float(np.std(dl2s, ddof=1)), 2),
            "dl2_range": [min(dl2s), max(dl2s)],
            "dtot_mean": round(float(np.mean(dtots)), 2),
            "dtot_sd": round(float(np.std(dtots, ddof=1)), 2),
            "dtot_range": [min(dtots), max(dtots)],
        },
        "e60_s60_drift_sim": e60_s60,
    }


def run_42_analyses() -> dict:
    d2 = json.loads((BUILD_DIR / "deep_rtl_validation_part2.json").read_text())
    d1 = json.loads((BUILD_DIR / "deep_rtl_validation_part1.json").read_text())
    d3 = json.loads((BUILD_DIR / "deep_rtl_validation_part3.json").read_text())
    oos = [r for r in d2["records"] if r["group"] == "oos_60pt"]

    wl_map = {s["name"]: make_synthetic_integer_ffn(s["d_model"], s["d_ff"], seed=20261130 + i) for i, s in enumerate(SHAPES_6)}

    # 1) Residual regression: res = silicon - model
    X_rows, y_res = [], []
    for r in oos:
        w = wl_map[r["shape"]]
        M_tot = w["l1_meta"]["n_tiles"] * w["l1_meta"]["k_chunks"] + w["l2_meta"]["n_tiles"] * w["l2_meta"]["k_chunks"]
        kch_max = max(w["l1_meta"]["k_chunks"], w["l2_meta"]["k_chunks"])
        res_cyc = r["meas_tot_mean"] - r["wb_pred"]
        X_rows.append([1.0, float(M_tot), float(r["sdram_mhz"]), float(kch_max)])
        y_res.append(res_cyc)
    X = np.array(X_rows); y = np.array(y_res)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    corr_err_pct = np.abs(y - y_hat) / np.array([r["meas_tot_mean"] for r in oos]) * 100.0
    raw_err_pct = np.array([r["wb_err_pct"] for r in oos])

    # 2) Separate 12 pilot-equivalent conditions (3 shapes x {60,90,100,120}) vs 48 unseen conditions
    pilot_shapes = {"ffn_192_384_192", "ffn_256_576_256", "ffn_256_960_256"}
    pilot_freqs = {60, 90, 100, 120}
    dev12 = [r for r in oos if r["shape"] in pilot_shapes and r["sdram_mhz"] in pilot_freqs]
    unseen48 = [r for r in oos if not (r["shape"] in pilot_shapes and r["sdram_mhz"] in pilot_freqs)]

    def stats_err(sub):
        e = [r["wb_err_pct"] for r in sub]
        return {
            "n": len(sub),
            "mean_pct": round(float(np.mean(e)), 5),
            "median_pct": round(float(np.median(e)), 5),
            "p95_pct": round(float(np.percentile(e, 95)), 5),
            "max_pct": round(float(np.max(e)), 5),
        }

    # 3) Non-parametric Sign / Wilcoxon + TOST on N=100 pairs (part1) & N=40 A+B pairs (region_swap)
    runs100 = d1["sdram_90mhz"]["phase_100pairs"]["runs"]
    o1 = [r for r in runs100 if r["offset"] == 1]
    o15 = [r for r in runs100 if r["offset"] == 15]
    dt100 = np.array([a["tot"] - b["tot"] for a, b in zip(o1, o15)], dtype=np.float64)
    dl2_100 = np.array([a["l2_tot"] - b["l2_tot"] for a, b in zip(o1, o15)], dtype=np.float64)
    # Two-sided sign test & TOST around +42 on L2: H0: |mean(dl2) - 42| >= delta
    mean_l2 = float(np.mean(dl2_100))
    se_l2 = float(np.std(dl2_100, ddof=1) / np.sqrt(len(dl2_100)))
    # Check delta = 3 cycles (i.e. [39, 45]):
    t_lo = (mean_l2 - 39.0) / se_l2
    t_hi = (45.0 - mean_l2) / se_l2

    # 4) Average-page vs Per-packet on 0..31 sweep (n=10, 90 MHz)
    sw = d3["offset_0_31_n10_90mhz"]["rows"]
    # Average-page: all non-%8 offsets get the mean of non-%8 predictions; %8 offsets get %8 prediction
    # Better: constant average-page across ALL 32 offsets uses mean(p_i) = 4.25 (L1), 6.25 (L2)
    # Even stricter: within non-aligned offsets (off % 8 != 0), average-page gives constant 65265 while per-packet tracks 65242..65292
    non8 = [x for x in sw if x["offset"] % 8 != 0]
    avg_page_const = float(np.mean([x["pred_tot"] for x in non8]))
    meas_non8 = np.array([x["meas_tot_mean"] for x in non8])
    pp_non8 = np.array([x["pred_tot"] for x in non8])
    mae_pp_non8 = float(np.mean(np.abs(meas_non8 - pp_non8)))
    mae_avg_non8 = float(np.mean(np.abs(meas_non8 - avg_page_const)))
    # Also on all 32 offsets vs constant mean(pred)
    all_meas = np.array([x["meas_tot_mean"] for x in sw])
    all_pp = np.array([x["pred_tot"] for x in sw])
    mae_pp_all = float(np.mean(np.abs(all_meas - all_pp)))
    mae_avg_all = float(np.mean(np.abs(all_meas - float(np.mean(all_pp)))))

    # 5) Clock selection sensitivity across {0.5%, 1.0%, 2.0%} on 10 frequencies
    freqs = [60, 70, 75, 80, 88, 90, 96, 100, 110, 120]
    sens = {}
    for thr in (0.005, 0.01, 0.02):
        wb_match, bist_match = 0, 0
        details = []
        for s in SHAPES_6:
            sn = s["name"]
            sub = {r["sdram_mhz"]: r for r in oos if r["shape"] == sn}
            f_m = min(f for f in freqs if sub[f]["meas_tot_mean"] <= (1.0 + thr) * sub[120]["meas_tot_mean"])
            f_w = min(f for f in freqs if sub[f]["wb_pred"] <= (1.0 + thr) * sub[120]["wb_pred"])
            f_b = min(f for f in freqs if sub[f]["bist_pred"] <= (1.0 + thr) * sub[120]["bist_pred"])
            wb_match += int(f_w == f_m)
            bist_match += int(f_b == f_m)
            details.append({"shape": sn, "sil": f_m, "wb": f_w, "bist": f_b})
        sens[f"thr_{thr*100:.1f}pct"] = {"wb_match": f"{wb_match}/6", "bist_match": f"{bist_match}/6", "details": details}

    return {
        "residual_regression_60pt": {
            "features": ["intercept", "M_packets", "f_sdram_mhz", "k_chunks_max"],
            "beta": [round(float(b), 4) for b in beta],
            "raw_mean_pct": round(float(np.mean(raw_err_pct)), 5),
            "raw_max_pct": round(float(np.max(raw_err_pct)), 5),
            "after_bias_corr_mean_pct": round(float(np.mean(corr_err_pct)), 5),
            "after_bias_corr_max_pct": round(float(np.max(corr_err_pct)), 5),
        },
        "dev12_vs_unseen48": {
            "dev12": stats_err(dev12),
            "unseen48": stats_err(unseen48),
        },
        "nonparam_and_tost_100pairs": {
            "n_pairs": len(dt100),
            "tot_pos_count": int(np.sum(dt100 > 0)),
            "l2_pos_count": int(np.sum(dl2_100 > 0)),
            "l2_median_cyc": float(np.median(dl2_100)),
            "l2_iqr_cyc": [float(np.percentile(dl2_100, 25)), float(np.percentile(dl2_100, 75))],
            "l2_mean_se": [round(mean_l2, 2), round(se_l2, 2)],
            "tost_delta_3cyc_t_lo_hi": [round(t_lo, 2), round(t_hi, 2)],
        },
        "dse_offset_0_31_per_pkt_vs_avg_page": {
            "non8_offsets_n": len(non8),
            "non8_mae_per_packet_cyc": round(mae_pp_non8, 2),
            "non8_mae_avg_page_cyc": round(mae_avg_non8, 2),
            "all32_mae_per_packet_cyc": round(mae_pp_all, 2),
            "all32_mae_avg_page_cyc": round(mae_avg_all, 2),
        },
        "clock_selection_sensitivity_10freq": sens,
    }


def main() -> int:
    res = {
        "sec_4_1": run_41_sims(),
        "sec_4_2": run_42_analyses(),
    }
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
