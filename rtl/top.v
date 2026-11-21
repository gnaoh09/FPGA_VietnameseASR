`timescale 1ns/1ps

// Top-level ZipFormer Cyclone IV Hardware Wrapper
// Clocks generated from 50 MHz PIN_23 PLL:
//   c0: 80 MHz -> jtag_bridge + zipformer_engine
//   c1: SDRAM_MHZ (80 or 100 MHz) -> sdram_ctrl
//   c2: SDRAM_MHZ (-3 ns phase shift) -> PIN_43 sdram_clk
module top #(
    parameter integer NUM_LANES  = 16,
    parameter integer ENG_MHZ    = 80,
    parameter integer SDRAM_MHZ  = 100,
    parameter integer MAX_T_TILE = 4
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [3:0]  key,
    output wire [3:0]  led,
    // External 8 MiB SDR SDRAM (HY57V641620FTP-H, U6)
    output wire        sdram_clk,
    output wire        sdram_cke,
    output wire        sdram_cs_n,
    output wire        sdram_ras_n,
    output wire        sdram_cas_n,
    output wire        sdram_we_n,
    output wire [1:0]  sdram_ba,
    output wire [11:0] sdram_addr,
    output wire [1:0]  sdram_dqm,
    inout  wire [15:0] sdram_dq
`ifdef SIMULATION
    ,input  wire       sim_tck,
    input  wire       sim_tms,
    input  wire       sim_tdi,
    output wire       sim_tdo
`endif
);
`ifdef SIMULATION
    localparam integer SDRAM_INIT_CYC = 200;
`else
    localparam integer SDRAM_INIT_CYC = (SDRAM_MHZ * 220);
`endif

    wire        clk_80m;
    wire        clk_sdram;
    wire        pll_locked;
    wire        sys_rst_n = rst_n & pll_locked;

    pll_80m #(
        .ENG_MHZ   (ENG_MHZ),
        .SDRAM_MHZ (SDRAM_MHZ)
    ) u_pll (
        .inclk0 (clk),
        .areset (~rst_n),
        .c0     (clk_80m),
        .c1     (clk_sdram),
        .c2     (sdram_clk),
        .locked (pll_locked)
    );

    wire        cmd_valid;
    wire [63:0] cmd_word;
    wire [63:0] resp_word;
    wire        status_busy;
    wire        status_done;

    // Engine <-> SDRAM CDC wires
    wire         sdram_req_tog;
    wire  [1:0]  sdram_req_op;
    wire [18:0]  sdram_req_addr;
    wire [19:0]  sdram_req_len;
    wire [127:0] sdram_req_wdata;
    wire  [2:0]  sdram_cfg_rd_lat;
    wire         sdram_ack_tog;
    wire [127:0] sdram_rdata;
    wire [31:0]  sdram_bist_wr_cyc;
    wire [31:0]  sdram_bist_rd_cyc;
    wire [31:0]  sdram_bist_err;
    wire         sdram_ready;
    wire         sdram_dma_w128_valid;
    wire [127:0] sdram_dma_w128_data;
    wire  [8:0]  sdram_dma_w128_idx;
    wire [63:0]  sdram_cyc_cnt;

    wire [15:0] sdram_dq_out;
    wire        sdram_dq_oe;
    assign sdram_dq = sdram_dq_oe ? sdram_dq_out : 16'hZZZZ;

`ifndef SIMULATION
    wire sim_tck = 1'b0;
    wire sim_tms = 1'b0;
    wire sim_tdi = 1'b0;
    wire sim_tdo;
`endif

    jtag_bridge u_jtag (
        .clk       (clk_80m),
        .rst_n     (sys_rst_n),
        .sim_tck   (sim_tck),
        .sim_tms   (sim_tms),
        .sim_tdi   (sim_tdi),
        .sim_tdo   (sim_tdo),
        .cmd_valid (cmd_valid),
        .cmd_word  (cmd_word),
        .resp_word (resp_word)
    );

    sdram_ctrl #(
        .CLK_MHZ       (SDRAM_MHZ),
        .INIT_WAIT_CYC (SDRAM_INIT_CYC)
    ) u_sdram (
        .clk_sdram          (clk_sdram),
        .clk_eng            (clk_80m),
        .rst_n              (sys_rst_n),
        .eng_req_tog        (sdram_req_tog),
        .eng_req_op         (sdram_req_op),
        .eng_addr           (sdram_req_addr),
        .eng_len            (sdram_req_len),
        .eng_wdata          (sdram_req_wdata),
        .eng_rd_lat         (sdram_cfg_rd_lat),
        .eng_ack_tog        (sdram_ack_tog),
        .eng_rdata          (sdram_rdata),
        .eng_bist_wr_cyc    (sdram_bist_wr_cyc),
        .eng_bist_rd_cyc    (sdram_bist_rd_cyc),
        .eng_bist_err       (sdram_bist_err),
        .eng_sdram_ready    (sdram_ready),
        .eng_dma_w128_valid (sdram_dma_w128_valid),
        .eng_dma_w128_data  (sdram_dma_w128_data),
        .eng_dma_w128_idx   (sdram_dma_w128_idx),
        .eng_sdram_cyc_cnt  (sdram_cyc_cnt),
        .sdram_cke          (sdram_cke),
        .sdram_cs_n         (sdram_cs_n),
        .sdram_ras_n        (sdram_ras_n),
        .sdram_cas_n        (sdram_cas_n),
        .sdram_we_n         (sdram_we_n),
        .sdram_ba           (sdram_ba),
        .sdram_addr         (sdram_addr),
        .sdram_dqm          (sdram_dqm),
        .sdram_dq_in        (sdram_dq),
        .sdram_dq_out       (sdram_dq_out),
        .sdram_dq_oe        (sdram_dq_oe)
    );

    zipformer_engine #(
        .NUM_LANES  (NUM_LANES),
        .MAX_T_TILE (MAX_T_TILE)
    ) u_engine (
        .clk                  (clk_80m),
        .rst_n                (sys_rst_n),
        .cmd_valid            (cmd_valid),
        .cmd_word             (cmd_word),
        .resp_word            (resp_word),
        .status_busy          (status_busy),
        .status_done          (status_done),
        .sdram_req_tog        (sdram_req_tog),
        .sdram_req_op         (sdram_req_op),
        .sdram_req_addr       (sdram_req_addr),
        .sdram_req_len        (sdram_req_len),
        .sdram_req_wdata      (sdram_req_wdata),
        .sdram_cfg_rd_lat     (sdram_cfg_rd_lat),
        .sdram_ack_tog        (sdram_ack_tog),
        .sdram_rdata          (sdram_rdata),
        .sdram_bist_wr_cyc    (sdram_bist_wr_cyc),
        .sdram_bist_rd_cyc    (sdram_bist_rd_cyc),
        .sdram_bist_err       (sdram_bist_err),
        .sdram_ready          (sdram_ready),
        .sdram_dma_w128_valid (sdram_dma_w128_valid),
        .sdram_dma_w128_data  (sdram_dma_w128_data),
        .sdram_dma_w128_idx   (sdram_dma_w128_idx),
        .sdram_cyc_cnt        (sdram_cyc_cnt)
    );

    // Heartbeat (80 MHz -> ~1.2 Hz on bit 25), SDRAM ready & JTAG activity on 4 active-low LEDs
    reg [25:0] hb_cnt;
    reg        jtag_seen;

    always @(posedge clk_80m or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            hb_cnt    <= 26'd0;
            jtag_seen <= 1'b0;
        end else begin
            hb_cnt <= hb_cnt + 26'd1;
            if (cmd_valid) begin
                jtag_seen <= 1'b1;
            end
        end
    end

    wire lamp_test = ~key[3];
    wire unused_keys = &{1'b0, key[2:0]};
    wire [3:0] led_int = {jtag_seen | unused_keys, status_done | sdram_ready, status_busy | pll_locked, hb_cnt[25]};
    assign led = lamp_test ? 4'b0000 : ~led_int;
endmodule
