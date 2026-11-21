`timescale 1ns/1ps

// ZipFormer Integer Accelerator Engine for Cyclone IV
// Supports:
//   - Parameterized NUM_LANES (16 for EP4CE6E22C8, 32 for EP4CE10E22C8) INT8 MAC lanes
//   - Hardware W4 (packed 2x signed int4 per byte) -> INT8 decompressor
//   - Pipelined Epilogue: int32 bias + dyadic M16 * 2^-S (round-half-up) + saturate [-128,127] + 256-entry LUT (SwooshL / Tanh)
//   - K-tiling partial accumulation (psum_mem 64x32-bit) & 2D act/dst stride for full 512->1536->512 FFN streaming
//   - Mode V (single layer) & Mode S (chained 2-layer FFN or Joiner Add+Tanh+Linear)
//   - Hardware continuous stress_loop mode for external power measurement
module zipformer_engine #(
    parameter integer NUM_LANES  = 16,
    parameter integer MAX_T_TILE = 4
) (
    input  wire         clk,
    input  wire         rst_n,
    // JTAG command interface
    input  wire         cmd_valid,
    input  wire  [63:0] cmd_word,
    output reg   [63:0] resp_word,
    // Status LEDs (active-high internal)
    output wire         status_busy,
    output wire         status_done,
    // SDRAM controller CDC interface
    output reg          sdram_req_tog,
    output reg   [1:0]  sdram_req_op,
    output reg  [18:0]  sdram_req_addr,
    output reg  [19:0]  sdram_req_len,
    output reg [127:0]  sdram_req_wdata,
    output wire  [2:0]  sdram_cfg_rd_lat,
    input  wire         sdram_ack_tog,
    input  wire [127:0] sdram_rdata,
    input  wire [31:0]  sdram_bist_wr_cyc,
    input  wire [31:0]  sdram_bist_rd_cyc,
    input  wire [31:0]  sdram_bist_err,
    input  wire         sdram_ready,
    input  wire         sdram_dma_w128_valid,
    input  wire [127:0] sdram_dma_w128_data,
    input  wire  [8:0]  sdram_dma_w128_idx,
    input  wire [63:0]  sdram_cyc_cnt
);
    // JTAG opcodes
    localparam [7:0]
        OP_STAGE_W32    = 8'h01,
        OP_COMMIT_ACT   = 8'h02,
        OP_COMMIT_WGT   = 8'h03,
        OP_WRITE_PARAM  = 8'h04,
        OP_WRITE_LUT    = 8'h05,
        OP_SET_READ_PTR = 8'h06,
        OP_WRITE_REG    = 8'h07,
        OP_START        = 8'h08,
        OP_SDRAM_CMD    = 8'h09;

    // FSM states
    localparam [3:0]
        ST_IDLE        = 4'd0,
        ST_JOINER_RENC = 4'd1,
        ST_JOINER_WENC = 4'd2,
        ST_JOINER_RDEC = 4'd3,
        ST_JOINER_WDEC = 4'd4,
        ST_JOINER_LUT  = 4'd5,
        ST_GEMM_RUN    = 4'd6,
        ST_GEMM_DRAIN  = 4'd7,
        ST_WAIT_DMA    = 4'd8;

    /* verilator lint_off UNUSEDSIGNAL */
    wire [7:0]  cmd_op    = cmd_word[63:56];
    wire [23:0] cmd_addr  = cmd_word[55:32];
    wire [31:0] cmd_wdata = cmd_word[31:0];

    localparam integer ACT_AW     = (MAX_T_TILE > 4) ? 10 : 9;
    localparam integer ACT_DEPTH  = (1 << ACT_AW);
    localparam integer OUT_AW     = (MAX_T_TILE > 4) ? 8 : 7;
    localparam integer OUT_DEPTH  = (1 << OUT_AW);
    localparam integer PSUM_AW    = (MAX_T_TILE > 4) ? 7 : 6;
    localparam integer PSUM_DEPTH = (1 << PSUM_AW);

    // Configuration registers
    reg [31:0] reg_cfg0;
    reg [31:0] reg_cfg1;
    reg [31:0] reg_cfg2;
    reg [31:0] reg_cfg3;
    reg [31:0] reg_cfg4;
    reg [31:0] reg_cfg5;

    wire [1:0] cfg_op_mode    = reg_cfg0[1:0];   // 0=Single, 1=FFN Chain (Mode S), 2=Joiner Chain (Mode S)
    wire       cfg_w4_mode    = reg_cfg0[2];     // 1=packed INT4 weights
    wire [1:0] cfg_lut_mode   = reg_cfg0[4:3];   // 0=none, 1=SwooshL, 2=Tanh
    wire [7:0] cfg_dim_T      = reg_cfg0[12:5];
    wire [9:0] cfg_dim_K      = {reg_cfg3[9:8], reg_cfg0[20:13]};
    wire [7:0] cfg_dim_N      = reg_cfg0[28:21];
    wire [1:0] cfg_acc_mode   = reg_cfg0[30:29]; // 0=normal, 1=store_psum, 2=load_store_psum, 3=load_epi

    wire [7:0] cfg_dim_N2     = reg_cfg1[7:0];
    wire [ACT_AW-1:0] cfg_act_base;
    wire [7:0] cfg_wgt_base   = reg_cfg1[23:16];
    wire [ACT_AW-1:0] cfg_hpost_base;

    wire [7:0] cfg_wgt2_base  = reg_cfg2[7:0];
    wire [6:0] cfg_param1_base= reg_cfg2[14:8];
    wire [6:0] cfg_param2_base= reg_cfg2[22:16];
    wire [ACT_AW-1:0] cfg_out_base;

    generate
        if (ACT_AW > 9) begin : gen_cfg_base_wide
            assign cfg_act_base   = {{(ACT_AW-9){1'b0}}, reg_cfg3[10], reg_cfg1[15:8]};
            assign cfg_hpost_base = {{(ACT_AW-9){1'b0}}, reg_cfg3[11], reg_cfg1[31:24]};
            assign cfg_out_base   = {{(ACT_AW-8){1'b0}}, reg_cfg2[31:24]};
        end else begin : gen_cfg_base_narrow
            assign cfg_act_base   = {reg_cfg3[10], reg_cfg1[15:8]};
            assign cfg_hpost_base = {reg_cfg3[11], reg_cfg1[31:24]};
            assign cfg_out_base   = {1'b0, reg_cfg2[31:24]};
        end
    endgenerate

    wire [7:0] cfg_dec_base    = reg_cfg3[7:0];
    wire [7:0] cfg_dst_stride  = (reg_cfg3[19:12] == 8'd0) ? 8'd1 : reg_cfg3[19:12];
    wire       cfg_stress_loop = reg_cfg3[20];
    wire [7:0] cfg_act_stride  = reg_cfg3[28:21];
    wire       cfg_dst_is_act  = reg_cfg3[29];

    // SDRAM streaming configuration (reg_cfg4, reg_cfg5)
    wire        cfg_sdram_stream_en = reg_cfg4[0];
    wire [7:0]  cfg_sdram_ntiles    = reg_cfg4[8:1];
    wire [3:0]  cfg_sdram_kchunks   = reg_cfg4[12:9];
    wire [2:0]  cfg_sdram_rd_lat    = (reg_cfg4[15:13] == 3'd0) ? 3'd4 : reg_cfg4[15:13];
    wire [8:0]  cfg_sdram_w_words   = reg_cfg4[24:16];
    wire [18:0] cfg_sdram_base      = reg_cfg5[18:0];
    /* verilator lint_on UNUSEDSIGNAL */

    assign sdram_cfg_rd_lat = cfg_sdram_rd_lat;

    // Staging 128-bit register for JTAG loads
    reg [127:0] stage128;

    // Performance & status registers
    reg [3:0]  state;
    reg        done_sticky;
    reg        err_flag;
    reg [31:0] hw_total_cycles;
    reg [31:0] hw_mac_cycles;
    reg [31:0] hw_epilogue_cycles;
    reg [31:0] hw_wgt_reads;
    reg [31:0] hw_run_count;
    reg [31:0] hw_out_checksum;
    reg [31:0] hw_sdram_stall_cyc;
    reg [31:0] hw_dma_cyc_cnt;
    reg [63:0] free_clk_cnt;

    assign status_busy = (state != ST_IDLE);
    assign status_done = done_sticky;

    // -------------------------------------------------------------------------
    // M9K Memory Subsystems (strictly synchronous 1R + 1W for clean M9K infer)
    //   act_mem: 512 x 128-bit (8 M9K on 16-lane) -> holds X[4x512] + h_post[4x1536]
    //   wgt_mem: 256 x 128-bit (4 M9K) -> 16 rows x 512 cols in W4 or 16x256 in W8
    //   out_mem: 128 x 128-bit (4 M9K)
    //   param_mem: 128 x 64-bit (2 M9K)
    //   lut_mem: 512 x 8-bit (1 M9K)
    //   psum_mem: 64 x 32-bit (1 M9K) -> int32 partial sums for K-tiling
    // -------------------------------------------------------------------------
    reg         act_we;
    reg  [ACT_AW-1:0]  act_waddr;
    reg [127:0] act_wdata;
    reg  [ACT_AW-1:0]  act_raddr;
    wire [(NUM_LANES*8)-1:0] act_rdata_wide;
    wire [127:0] act_rdata128;

    reg         wgt_we;
    reg  [8:0]  wgt_waddr;
    reg [127:0] wgt_wdata;
    reg  [8:0]  wgt_raddr;
    wire [(NUM_LANES*8)-1:0] wgt_rdata_wide;
    wire [127:0] wgt_rdata128;

    reg         out_we;
    reg  [OUT_AW-1:0]  out_waddr;
    reg [127:0] out_wdata;
    reg  [OUT_AW-1:0]  out_raddr;
    reg [127:0] out_rdata;

    reg         param_we;
    reg  [6:0]  param_waddr;
    reg  [63:0] param_wdata;
    reg  [6:0]  param_raddr;
    /* verilator lint_off UNUSEDSIGNAL */
    reg  [63:0] param_rdata;
    /* verilator lint_on UNUSEDSIGNAL */

    reg         lut_we;
    reg  [8:0]  lut_waddr;
    reg  [7:0]  lut_wdata;
    reg  [8:0]  lut_raddr;
    reg  [7:0]  lut_rdata;

    reg         psum_we;
    reg  [PSUM_AW-1:0]  psum_waddr;
    reg signed [31:0] psum_wdata;
    reg  [PSUM_AW-1:0]  psum_raddr;
    reg signed [31:0] psum_rdata;

    generate
        if (NUM_LANES == 16) begin : gen_mem_16
            (* ramstyle = "M9K, no_rw_check" *) reg [127:0] act_mem [0:ACT_DEPTH-1];
            (* ramstyle = "M9K, no_rw_check" *) reg [127:0] wgt_mem [0:511];
            reg [127:0] act_q;
            reg [127:0] wgt_q;
            always @(posedge clk) begin
                if (act_we) act_mem[act_waddr] <= act_wdata;
                act_q <= act_mem[act_raddr];
            end
            always @(posedge clk) begin
                if (wgt_we) wgt_mem[wgt_waddr] <= wgt_wdata;
                wgt_q <= wgt_mem[wgt_raddr];
            end
            assign act_rdata_wide = act_q;
            assign act_rdata128   = act_q;
            assign wgt_rdata_wide = wgt_q;
            assign wgt_rdata128   = wgt_q;
        end else begin : gen_mem_32
            (* ramstyle = "M9K, no_rw_check" *) reg [127:0] act_b0 [0:255];
            (* ramstyle = "M9K, no_rw_check" *) reg [127:0] act_b1 [0:255];
            (* ramstyle = "M9K, no_rw_check" *) reg [127:0] wgt_b0 [0:255];
            (* ramstyle = "M9K, no_rw_check" *) reg [127:0] wgt_b1 [0:255];
            reg [127:0] act_q0, act_q1;
            reg [127:0] wgt_q0, wgt_q1;
            reg         act_rsel_d, wgt_rsel_d;

            always @(posedge clk) begin
                if (act_we && !act_waddr[0]) act_b0[act_waddr[8:1]] <= act_wdata;
                if (act_we &&  act_waddr[0]) act_b1[act_waddr[8:1]] <= act_wdata;
                act_q0     <= act_b0[act_raddr[8:1]];
                act_q1     <= act_b1[act_raddr[8:1]];
                act_rsel_d <= act_raddr[0];
            end
            always @(posedge clk) begin
                if (wgt_we && !wgt_waddr[0]) wgt_b0[wgt_waddr[8:1]] <= wgt_wdata;
                if (wgt_we &&  wgt_waddr[0]) wgt_b1[wgt_waddr[8:1]] <= wgt_wdata;
                wgt_q0     <= wgt_b0[wgt_raddr[8:1]];
                wgt_q1     <= wgt_b1[wgt_raddr[8:1]];
                wgt_rsel_d <= wgt_raddr[0];
            end
            assign act_rdata_wide = {act_q1, act_q0};
            assign act_rdata128   = act_rsel_d ? act_q1 : act_q0;
            assign wgt_rdata_wide = {wgt_q1, wgt_q0};
            assign wgt_rdata128   = wgt_rsel_d ? wgt_q1 : wgt_q0;
        end
    endgenerate

    (* ramstyle = "M9K, no_rw_check" *) reg [127:0] out_mem   [0:OUT_DEPTH-1];
    (* ramstyle = "M9K, no_rw_check" *) reg  [63:0] param_mem [0:127];
    (* ramstyle = "M9K, no_rw_check" *) reg   [7:0] lut_mem   [0:511];
    (* ramstyle = "M9K, no_rw_check" *) reg signed [31:0] psum_mem [0:PSUM_DEPTH-1];

    always @(posedge clk) begin
        if (out_we) out_mem[out_waddr] <= out_wdata;
        out_rdata <= out_mem[out_raddr];
    end

    always @(posedge clk) begin
        if (param_we) param_mem[param_waddr] <= param_wdata;
        param_rdata <= param_mem[param_raddr];
    end

    always @(posedge clk) begin
        if (lut_we) lut_mem[lut_waddr] <= lut_wdata;
        lut_rdata <= lut_mem[lut_raddr];
    end

    always @(posedge clk) begin
        if (psum_we) psum_mem[psum_waddr] <= psum_wdata;
        psum_rdata <= psum_mem[psum_raddr];
    end

    // -------------------------------------------------------------------------
    // Active Pass Controls & GEMM/Joiner Counters
    // -------------------------------------------------------------------------
    reg        pass_idx;
    reg  [7:0] cur_T;
    reg  [5:0] cur_k16;
    reg  [3:0] cur_n16;
    reg  [ACT_AW-1:0] cur_act_base;
    reg  [8:0] cur_wgt_base;
    reg  [6:0] cur_param_base;
    reg        cur_dst_is_act;
    reg  [1:0] cur_lut_mode;
    reg  [1:0] cur_acc_mode;
    reg  [7:0] cur_dst_stride;
    reg  [7:0] cur_act_stride;

    // SDRAM Ping-Pong Stream & JTAG pointer registers
    reg [18:0] jtag_sdram_ptr;
    reg  [1:0] jtag_sdram_auto_sub;
    reg        dma_in_flight;
    reg        dma_bank;
    reg        cur_comp_bank;
    reg        dma_all_issued;
    reg  [7:0] dma_nt;
    reg  [3:0] dma_kc;
    reg [18:0] dma_next_addr;
    reg  [8:0] dma_words_rx;
    reg  [7:0] cur_nt;
    reg  [3:0] cur_kc_tile;
    reg  [ACT_AW-1:0] cur_kc_act_off;
    reg        dma_param_pending;
    reg [63:0] dma_param_hi_data;
    reg  [6:0] dma_param_hi_addr;

    wire       sdram_cmd_busy = (sdram_req_tog != sdram_ack_tog);
    wire [8:0] dma_req_words  = 9'd8 + cfg_sdram_w_words;

    reg        bank0_valid, bank1_valid;
    wire       cur_comp_bank_valid = cur_comp_bank ? bank1_valid : bank0_valid;
    wire       alt_comp_bank_valid = cur_comp_bank ? bank0_valid : bank1_valid;
    wire       cur_dma_bank_valid  = dma_bank      ? bank1_valid : bank0_valid;

    wire       stream_has_more_kc  = ({1'b0, cur_kc_tile} + 5'd1 < {1'b0, cfg_sdram_kchunks});
    wire [7:0] next_stream_nt      = stream_has_more_kc ? cur_nt : (cur_nt + 8'd1);
    wire [3:0] next_stream_kc      = stream_has_more_kc ? (cur_kc_tile + 4'd1) : 4'd0;
    wire [ACT_AW-1:0] next_stream_act_off = stream_has_more_kc ? (cur_kc_act_off + {{(ACT_AW-6){1'b0}}, cur_k16}) : {ACT_AW{1'b0}};
    wire       stream_last_chunk   = !stream_has_more_kc && (cur_nt + 8'd1 >= cfg_sdram_ntiles);

    function [1:0] fn_stream_acc_mode;
        input [3:0] kc_idx;
        input [3:0] k_total;
        input [1:0] base_mode;
        begin
            if (k_total <= 4'd1) fn_stream_acc_mode = base_mode;
            else if (kc_idx == 4'd0) fn_stream_acc_mode = 2'd1;
            else if ({1'b0, kc_idx} + 5'd1 < {1'b0, k_total}) fn_stream_acc_mode = 2'd2;
            else fn_stream_acc_mode = 2'd3;
        end
    endfunction

    wire [7:0] cur_k_chunks   = (NUM_LANES == 32) ? {3'd0, cur_k16[5:1]} : {2'd0, cur_k16};
    wire [7:0] cur_k16_chunks = {2'd0, cur_k16};
    wire [7:0] cur_n16_grps   = {4'd0, cur_n16};
    wire [7:0] eff_act_stride = (cur_act_stride == 8'd0) ? cur_k16_chunks : cur_act_stride;

    // GEMM issue counters
    reg  [7:0] g_t;
    reg  [ACT_AW-1:0] g_t_act_off;
    reg  [7:0] g_j_grp;
    reg  [3:0] g_j_sub;
    reg  [8:0] g_j_wgt_off;
    reg  [7:0] g_kc;

    // Joiner pass 0 registers
    reg  [7:0] jn_t;
    reg  [ACT_AW-1:0] jn_t_off;
    reg  [7:0] jn_kc;
    reg  [3:0] jn_lane;
    reg  [1:0] jn_wait;
    reg [127:0] jn_enc_buf;
    reg signed [7:0] jn_clamped [0:15];
    reg  [7:0] jn_pack [0:15];

    // Registered 2-cycle BRAM read capture for JTAG USER1 readback (no cmd_word->BRAM path)
    reg  [1:0] jtag_read_pending;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [15:0] jtag_read_sel;
    /* verilator lint_on UNUSEDSIGNAL */

    // -------------------------------------------------------------------------
    // Pipelined GEMM + Epilogue Datapath (Stages s0 -> s2..s5 + epi0..epi5)
    // -------------------------------------------------------------------------
    reg        s0_v, s0_first_kc, s0_last_kc, s0_w4_hi;
    reg  [3:0] s0_j_sub;
    reg  [PSUM_AW-1:0] s0_elem_idx;

    reg        s2_v, s2_first_kc, s2_last_kc;
    reg  [3:0] s2_j_sub;
    reg  [PSUM_AW-1:0] s2_elem_idx;
    reg signed [31:0] s2_bias;
    reg        [15:0] s2_m16;
    reg         [4:0] s2_shift;
    reg signed  [7:0] s2_a [0:NUM_LANES-1];
    reg signed  [7:0] s2_w [0:NUM_LANES-1];

    reg        s3_v, s3_first_kc, s3_last_kc;
    reg  [3:0] s3_j_sub;
    reg  [PSUM_AW-1:0] s3_elem_idx;
    reg signed [31:0] s3_bias;
    reg        [15:0] s3_m16;
    reg         [4:0] s3_shift;
    (* multstyle = "dsp" *) reg signed [15:0] s3_prod [0:NUM_LANES-1];

    reg        s4_v, s4_first_kc, s4_last_kc;
    reg  [3:0] s4_j_sub;
    reg  [PSUM_AW-1:0] s4_elem_idx;
    reg signed [31:0] s4_bias;
    reg        [15:0] s4_m16;
    reg         [4:0] s4_shift;
    reg signed [19:0] s4_psum [0:3];

    reg        s5_v, s5_first_kc, s5_last_kc;
    reg  [3:0] s5_j_sub;
    reg  [PSUM_AW-1:0] s5_elem_idx;
    reg signed [31:0] s5_bias;
    reg signed [31:0] s5_psum_init;
    reg        [15:0] s5_m16;
    reg         [4:0] s5_shift;
    reg signed [20:0] s5_sum;

    reg signed [31:0] acc_reg;
    reg        epi0_v;
    reg signed [31:0] epi0_acc;
    reg        [15:0] epi0_m16;
    reg         [4:0] epi0_shift;
    reg  [3:0] epi0_j_sub;

    reg        epi1_v;
    (* multstyle = "dsp" *) reg signed [47:0] epi1_prod;
    reg         [4:0] epi1_shift;
    reg  [3:0] epi1_j_sub;

    reg        epi2_v;
    reg signed [47:0] epi2_rounded;
    reg         [4:0] epi2_shift;
    reg  [3:0] epi2_j_sub;

    reg        epi2b_v;
    reg signed [47:0] epi2b_shifted;
    reg  [3:0] epi2b_j_sub;

    reg        epi3_v;
    reg signed [7:0] epi3_q8;
    reg  [3:0] epi3_j_sub;

    reg        epi4_v;
    reg signed [7:0] epi4_q8;
    reg  [3:0] epi4_j_sub;

    reg        epi5_v;
    reg  [ACT_AW-1:0] epi_dst_waddr;
    reg  [7:0] out_pack [0:15];

    wire pipeline_active = s0_v | s2_v | s3_v | s4_v | s5_v |
                           epi0_v | epi1_v | epi2_v | epi2b_v | epi3_v | epi4_v | epi5_v;

    // -------------------------------------------------------------------------
    // Pure Combinational Helper Block
    // -------------------------------------------------------------------------
    integer li;
    reg [7:0]         w4_nibble_byte;
    reg [3:0]         w4_nib;
    reg signed [7:0]  s2_w_comb [0:NUM_LANES-1];
    reg signed [19:0] psum_tmp0, psum_tmp1, psum_tmp2, psum_tmp3;
    reg signed [20:0] s4_sum_all;
    reg signed [31:0] acc_base;
    reg signed [31:0] acc_next;
    reg signed [47:0] round_const;
    reg signed [47:0] shifted_val;
    reg signed  [7:0] clamped_q8;
    reg signed  [8:0] join_sum9;
    reg signed  [7:0] join_clamp_comb [0:15];
    reg         [7:0] final_byte;
    reg       [127:0] commit_word128;
    wire              store_psum_only = (cur_acc_mode == 2'd1) || (cur_acc_mode == 2'd2);

    always @(*) begin
        for (li = 0; li < NUM_LANES; li = li + 1) begin
            if (!cfg_w4_mode) begin
                w4_nibble_byte = 8'd0;
                w4_nib         = 4'd0;
                s2_w_comb[li]  = wgt_rdata_wide[li*8 +: 8];
            end else begin
                if (NUM_LANES == 16) begin
                    w4_nibble_byte = s0_w4_hi ? wgt_rdata128[64 + (li >> 1)*8 +: 8] : wgt_rdata128[(li >> 1)*8 +: 8];
                end else begin
                    w4_nibble_byte = wgt_rdata128[(li >> 1)*8 +: 8];
                end
                w4_nib        = li[0] ? w4_nibble_byte[7:4] : w4_nibble_byte[3:0];
                s2_w_comb[li] = {{4{w4_nib[3]}}, w4_nib};
            end
        end

        psum_tmp0 = 20'sd0;
        psum_tmp1 = 20'sd0;
        psum_tmp2 = 20'sd0;
        psum_tmp3 = 20'sd0;
        for (li = 0; li < (NUM_LANES / 4); li = li + 1) begin
            psum_tmp0 = psum_tmp0 + {{4{s3_prod[li][15]}}, s3_prod[li]};
            psum_tmp1 = psum_tmp1 + {{4{s3_prod[(NUM_LANES/4) + li][15]}}, s3_prod[(NUM_LANES/4) + li]};
            psum_tmp2 = psum_tmp2 + {{4{s3_prod[(NUM_LANES/2) + li][15]}}, s3_prod[(NUM_LANES/2) + li]};
            psum_tmp3 = psum_tmp3 + {{4{s3_prod[(3*NUM_LANES/4) + li][15]}}, s3_prod[(3*NUM_LANES/4) + li]};
        end

        s4_sum_all = {{1{s4_psum[0][19]}}, s4_psum[0]} + {{1{s4_psum[1][19]}}, s4_psum[1]} +
                     {{1{s4_psum[2][19]}}, s4_psum[2]} + {{1{s4_psum[3][19]}}, s4_psum[3]};
        acc_base   = s5_first_kc ? (cur_acc_mode[1] ? s5_psum_init : s5_bias) : acc_reg;
        acc_next   = acc_base + {{11{s5_sum[20]}}, s5_sum};

        round_const = (epi1_shift == 5'd0) ? 48'sd0 : (48'sd1 <<< (epi1_shift - 5'd1));
        shifted_val = epi2_rounded >>> epi2_shift;
        if (epi2b_shifted > 48'sd127)       clamped_q8 = 8'sd127;
        else if (epi2b_shifted < -48'sd128) clamped_q8 = 8'sh80;
        else                                clamped_q8 = epi2b_shifted[7:0];

        join_sum9 = 9'sd0;
        for (li = 0; li < 16; li = li + 1) begin
            join_sum9 = $signed(jn_enc_buf[li*8 +: 8]) + $signed(act_rdata128[li*8 +: 8]);
            if (join_sum9 > 9'sd127)       join_clamp_comb[li] = 8'sd127;
            else if (join_sum9 < -9'sd128) join_clamp_comb[li] = 8'sh80;
            else                           join_clamp_comb[li] = join_sum9[7:0];
        end

        act_we      = 1'b0;
        act_waddr   = {ACT_AW{1'b0}};
        act_wdata   = 128'd0;
        act_raddr   = {ACT_AW{1'b0}};
        wgt_we      = 1'b0;
        wgt_waddr   = 9'd0;
        wgt_wdata   = 128'd0;
        wgt_raddr   = 9'd0;
        out_we      = 1'b0;
        out_waddr   = {OUT_AW{1'b0}};
        out_wdata   = 128'd0;
        out_raddr   = {OUT_AW{1'b0}};
        param_we    = 1'b0;
        param_waddr = 7'd0;
        param_wdata = 64'd0;
        param_raddr = 7'd0;
        lut_we      = 1'b0;
        lut_waddr   = 9'd0;
        lut_wdata   = 8'd0;
        lut_raddr   = 9'd0;
        psum_we     = 1'b0;
        psum_waddr  = {PSUM_AW{1'b0}};
        psum_wdata  = 32'sd0;
        psum_raddr  = s3_elem_idx;

        final_byte = (cur_lut_mode != 2'd0) ? lut_rdata : epi4_q8;
        commit_word128 = {
            final_byte,
            out_pack[14], out_pack[13], out_pack[12], out_pack[11],
            out_pack[10], out_pack[9],  out_pack[8],  out_pack[7],
            out_pack[6],  out_pack[5],  out_pack[4],  out_pack[3],
            out_pack[2],  out_pack[1],  out_pack[0]
        };

        if (state == ST_IDLE) begin
            if (cmd_valid && cmd_op == OP_COMMIT_ACT) begin
                act_we    = 1'b1;
                act_waddr = cmd_addr[ACT_AW-1:0];
                act_wdata = stage128;
            end
            if (cmd_valid && cmd_op == OP_COMMIT_WGT) begin
                wgt_we    = 1'b1;
                wgt_waddr = {1'b0, cmd_addr[7:0]};
                wgt_wdata = stage128;
            end
            if (cmd_valid && cmd_op == OP_WRITE_PARAM) begin
                param_we    = 1'b1;
                param_waddr = cmd_addr[6:0];
                param_wdata = stage128[63:0];
            end
            if (cmd_valid && cmd_op == OP_WRITE_LUT) begin
                lut_we    = 1'b1;
                lut_waddr = cmd_addr[8:0];
                lut_wdata = cmd_wdata[7:0];
            end
            // Registered read selector (zero cmd_word -> BRAM combinational delay)
            out_raddr = jtag_read_sel[OUT_AW-1:0];
            act_raddr = {jtag_read_sel[9 +: (ACT_AW-8)], jtag_read_sel[7:0]};
        end else begin
            // Background SDRAM DMA receiver writes strictly to inactive dma_bank
            if (sdram_dma_w128_valid) begin
                if (sdram_dma_w128_idx < 9'd8) begin
                    param_we    = 1'b1;
                    param_waddr = (dma_bank ? 7'd16 : 7'd0) + {3'd0, sdram_dma_w128_idx[2:0], 1'b0};
                    param_wdata = sdram_dma_w128_data[63:0];
                end else begin
                    wgt_we      = 1'b1;
                    wgt_waddr   = {dma_bank, sdram_dma_w128_idx[7:0] - 8'd8};
                    wgt_wdata   = sdram_dma_w128_data;
                end
            end else if (dma_param_pending) begin
                param_we    = 1'b1;
                param_waddr = dma_param_hi_addr;
                param_wdata = dma_param_hi_data;
            end

            if (state == ST_JOINER_RENC || state == ST_JOINER_WENC) begin
                act_raddr = cur_act_base + jn_t_off + {1'b0, jn_kc};
            end else if (state == ST_JOINER_RDEC || state == ST_JOINER_WDEC) begin
                act_raddr = {1'b0, cfg_dec_base} + {1'b0, jn_kc};
            end else if (state == ST_JOINER_LUT) begin
                lut_raddr = {1'b1, jn_clamped[jn_lane]};
                if (jn_wait == 2'd3) begin
                    act_we    = 1'b1;
                    act_waddr = cfg_hpost_base + jn_t_off + {1'b0, jn_kc};
                    act_wdata = {
                        jn_pack[15], jn_pack[14], jn_pack[13], jn_pack[12],
                        jn_pack[11], jn_pack[10], jn_pack[9],  jn_pack[8],
                        jn_pack[7],  jn_pack[6],  jn_pack[5],  jn_pack[4],
                        jn_pack[3],  jn_pack[2],  jn_pack[1],  jn_pack[0]
                    };
                end
            end else if (state == ST_GEMM_RUN || state == ST_GEMM_DRAIN) begin
                if (NUM_LANES == 16) begin
                    act_raddr = cur_act_base + g_t_act_off + {1'b0, g_kc};
                    wgt_raddr = cur_wgt_base + (cfg_w4_mode ? ({1'b0, g_j_wgt_off[8:1]} + {2'd0, g_kc[7:1]}) : ({1'b0, g_j_wgt_off[7:0]} + {1'b0, g_kc}));
                end else begin
                    act_raddr = cur_act_base + g_t_act_off + {1'b0, g_kc[6:0], 1'b0};
                    if (cfg_w4_mode) begin
                        wgt_raddr = cur_wgt_base + {1'b0, g_j_wgt_off[7:0]} + {1'b0, g_kc};
                    end else begin
                        wgt_raddr = cur_wgt_base + {1'b0, g_j_wgt_off[6:0], 1'b0} + {1'b0, g_kc[6:0], 1'b0};
                    end
                end
                param_raddr = cur_param_base + {g_j_grp[2:0], g_j_sub};
                lut_raddr   = {cur_lut_mode[1], epi3_q8};

                if (s5_v && s5_last_kc && store_psum_only) begin
                    psum_we    = 1'b1;
                    psum_waddr = s5_elem_idx;
                    psum_wdata = acc_next;
                end

                if (epi4_v && (epi4_j_sub == 4'd15)) begin
                    if (cur_dst_is_act) begin
                        act_we    = 1'b1;
                        act_waddr = epi_dst_waddr;
                        act_wdata = commit_word128;
                    end else begin
                        out_we    = 1'b1;
                        out_waddr = epi_dst_waddr[OUT_AW-1:0];
                        out_wdata = commit_word128;
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Main Sequential Datapath & Control
    // -------------------------------------------------------------------------
    integer si;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= ST_IDLE;
            reg_cfg0           <= 32'd0;
            reg_cfg1           <= 32'd0;
            reg_cfg2           <= 32'd0;
            reg_cfg3           <= 32'd0;
            reg_cfg4           <= 32'd0;
            reg_cfg5           <= 32'd0;
            stage128           <= 128'd0;
            done_sticky        <= 1'b0;
            err_flag           <= 1'b0;
            hw_total_cycles    <= 32'd0;
            hw_mac_cycles      <= 32'd0;
            hw_epilogue_cycles <= 32'd0;
            hw_wgt_reads       <= 32'd0;
            hw_run_count       <= 32'd0;
            hw_out_checksum    <= 32'd0;
            hw_sdram_stall_cyc <= 32'd0;
            hw_dma_cyc_cnt     <= 32'd0;
            free_clk_cnt       <= 64'd0;
            resp_word          <= 64'd0;
            jtag_read_pending  <= 2'd0;
            jtag_read_sel      <= 16'd0;

            sdram_req_tog      <= 1'b0;
            sdram_req_op       <= 2'd0;
            sdram_req_addr     <= 19'd0;
            sdram_req_len      <= 20'd0;
            sdram_req_wdata    <= 128'd0;
            jtag_sdram_ptr     <= 19'd0;
            jtag_sdram_auto_sub<= 2'd0;
            dma_in_flight      <= 1'b0;
            dma_bank           <= 1'b0;
            cur_comp_bank      <= 1'b0;
            dma_all_issued     <= 1'b0;
            dma_nt             <= 8'd0;
            dma_kc             <= 4'd0;
            dma_next_addr      <= 19'd0;
            dma_words_rx       <= 9'd0;
            cur_nt             <= 8'd0;
            cur_kc_tile        <= 4'd0;
            cur_kc_act_off     <= {ACT_AW{1'b0}};
            dma_param_pending  <= 1'b0;
            dma_param_hi_data  <= 64'd0;
            dma_param_hi_addr  <= 7'd0;
            bank0_valid        <= 1'b0;
            bank1_valid        <= 1'b0;

            pass_idx           <= 1'b0;
            cur_T              <= 8'd0;
            cur_k16            <= 6'd0;
            cur_n16            <= 4'd0;
            cur_act_base       <= {ACT_AW{1'b0}};
            cur_wgt_base       <= 9'd0;
            cur_param_base     <= 7'd0;
            cur_dst_is_act     <= 1'b0;
            cur_lut_mode       <= 2'd0;
            cur_acc_mode       <= 2'd0;
            cur_dst_stride     <= 8'd1;
            cur_act_stride     <= 8'd0;

            g_t                <= 8'd0;
            g_t_act_off        <= {ACT_AW{1'b0}};
            g_j_grp            <= 8'd0;
            g_j_sub            <= 4'd0;
            g_j_wgt_off        <= 9'd0;
            g_kc               <= 8'd0;

            jn_t               <= 8'd0;
            jn_t_off           <= {ACT_AW{1'b0}};
            jn_kc              <= 8'd0;
            jn_lane            <= 4'd0;
            jn_wait            <= 2'd0;
            jn_enc_buf         <= 128'd0;
            for (si = 0; si < 16; si = si + 1) begin
                jn_clamped[si] <= 8'sd0;
                jn_pack[si]    <= 8'd0;
                out_pack[si]   <= 8'd0;
            end

            s0_v <= 1'b0; s0_first_kc <= 1'b0; s0_last_kc <= 1'b0; s0_w4_hi <= 1'b0;
            s0_j_sub <= 4'd0; s0_elem_idx <= {PSUM_AW{1'b0}};
            s2_v <= 1'b0; s2_first_kc <= 1'b0; s2_last_kc <= 1'b0;
            s2_j_sub <= 4'd0; s2_elem_idx <= {PSUM_AW{1'b0}};
            s2_bias <= 32'sd0; s2_m16 <= 16'd0; s2_shift <= 5'd0;
            s3_v <= 1'b0; s3_first_kc <= 1'b0; s3_last_kc <= 1'b0;
            s3_j_sub <= 4'd0; s3_elem_idx <= {PSUM_AW{1'b0}};
            s3_bias <= 32'sd0; s3_m16 <= 16'd0; s3_shift <= 5'd0;
            s4_v <= 1'b0; s4_first_kc <= 1'b0; s4_last_kc <= 1'b0;
            s4_j_sub <= 4'd0; s4_elem_idx <= {PSUM_AW{1'b0}};
            s4_bias <= 32'sd0; s4_m16 <= 16'd0; s4_shift <= 5'd0;
            s5_v <= 1'b0; s5_first_kc <= 1'b0; s5_last_kc <= 1'b0;
            s5_j_sub <= 4'd0; s5_elem_idx <= {PSUM_AW{1'b0}};
            s5_bias <= 32'sd0; s5_psum_init <= 32'sd0; s5_m16 <= 16'd0; s5_shift <= 5'd0; s5_sum <= 21'sd0;
            for (si = 0; si < NUM_LANES; si = si + 1) begin
                s2_a[si]    <= 8'sd0;
                s2_w[si]    <= 8'sd0;
                s3_prod[si] <= 16'sd0;
            end
            for (si = 0; si < 4; si = si + 1) begin
                s4_psum[si] <= 20'sd0;
            end
            acc_reg <= 32'sd0;

            epi0_v <= 1'b0; epi0_acc <= 32'sd0; epi0_m16 <= 16'd0; epi0_shift <= 5'd0; epi0_j_sub <= 4'd0;
            epi1_v <= 1'b0; epi1_prod <= 48'sd0; epi1_shift <= 5'd0; epi1_j_sub <= 4'd0;
            epi2_v <= 1'b0; epi2_rounded <= 48'sd0; epi2_shift <= 5'd0; epi2_j_sub <= 4'd0;
            epi2b_v <= 1'b0; epi2b_shifted <= 48'sd0; epi2b_j_sub <= 4'd0;
            epi3_v <= 1'b0; epi3_q8 <= 8'sd0; epi3_j_sub <= 4'd0;
            epi4_v <= 1'b0; epi4_q8 <= 8'sd0; epi4_j_sub <= 4'd0;
            epi5_v <= 1'b0;
            epi_dst_waddr <= {ACT_AW{1'b0}};
        end else begin
            free_clk_cnt <= free_clk_cnt + 64'd1;

            s0_v   <= 1'b0;
            epi0_v <= 1'b0;
            epi5_v <= 1'b0;

            if (jtag_read_pending == 2'd2) begin
                jtag_read_pending <= 2'd1;
            end else if (jtag_read_pending == 2'd1) begin
                jtag_read_pending <= 2'd0;
                if (jtag_read_sel[15:12] == 4'h0) begin
                    resp_word <= jtag_read_sel[8] ? out_rdata[127:64] : out_rdata[63:0];
                end else if (jtag_read_sel[15:12] == 4'h1) begin
                    resp_word <= jtag_read_sel[8] ? act_rdata128[127:64] : act_rdata128[63:0];
                end
            end

            if (cmd_valid) begin
                case (cmd_op)
                    OP_STAGE_W32: begin
                        case (cmd_addr[1:0])
                            2'd0: stage128[31:0]   <= cmd_wdata;
                            2'd1: stage128[63:32]  <= cmd_wdata;
                            2'd2: stage128[95:64]  <= cmd_wdata;
                            2'd3: stage128[127:96] <= cmd_wdata;
                        endcase
                    end
                    OP_WRITE_REG: begin
                        case (cmd_addr[2:0])
                            3'd0: reg_cfg0 <= cmd_wdata;
                            3'd1: reg_cfg1 <= cmd_wdata;
                            3'd2: reg_cfg2 <= cmd_wdata;
                            3'd3: reg_cfg3 <= cmd_wdata;
                            3'd4: reg_cfg4 <= cmd_wdata;
                            3'd5: reg_cfg5 <= cmd_wdata;
                            default: ;
                        endcase
                    end
                    OP_SET_READ_PTR: begin
                        jtag_read_sel <= cmd_addr[15:0];
                        if (cmd_addr[15:12] == 4'h0 || cmd_addr[15:12] == 4'h1) begin
                            jtag_read_pending <= 2'd2;
                        end else if (cmd_addr[15:12] == 4'h2) begin
                            case (cmd_addr[3:0])
                                4'd0:  resp_word <= {32'h5A463445, 29'd0, err_flag, done_sticky, status_busy};
                                4'd1:  resp_word <= {hw_mac_cycles, hw_total_cycles};
                                4'd2:  resp_word <= {hw_wgt_reads, hw_epilogue_cycles};
                                4'd3:  resp_word <= {hw_run_count, hw_out_checksum};
                                4'd4:  resp_word <= free_clk_cnt;
                                4'd5:  resp_word <= {27'd0, sdram_cmd_busy, cfg_sdram_rd_lat, sdram_ready, NUM_LANES[15:0], 16'h0300}; // v3.0 (SDRAM + Ping-Pong DMA)
                                4'd6:  resp_word <= sdram_rdata[63:0];
                                4'd7:  resp_word <= sdram_rdata[127:64];
                                4'd8:  resp_word <= {sdram_bist_rd_cyc, sdram_bist_wr_cyc};
                                4'd9:  resp_word <= {hw_sdram_stall_cyc, sdram_bist_err};
                                4'd10: resp_word <= sdram_cyc_cnt;
                                4'd11: resp_word <= {hw_dma_cyc_cnt, 13'd0, jtag_sdram_ptr};
                                default: resp_word <= 64'hDEAD_BEEF_CAFE_BABE;
                            endcase
                        end else begin
                            resp_word <= stage128[63:0];
                        end
                    end
                    OP_SDRAM_CMD: begin
                        if (state == ST_IDLE) begin
                            case (cmd_addr[23:20])
                                4'h0: begin
                                    // SET_ADDR + reset auto sub-word counter
                                    jtag_sdram_ptr      <= cmd_wdata[18:0];
                                    jtag_sdram_auto_sub <= 2'd0;
                                end
                                4'h1: begin
                                    // PUSH_W32_AUTO: pack 4 x 32-bit -> auto OP_WR128 & ptr++
                                    stage128[(jtag_sdram_auto_sub * 32) +: 32] <= cmd_wdata;
                                    if (jtag_sdram_auto_sub != 2'd3) begin
                                        jtag_sdram_auto_sub <= jtag_sdram_auto_sub + 2'd1;
                                    end else begin
                                        jtag_sdram_auto_sub <= 2'd0;
                                        if (sdram_cmd_busy) begin
                                            err_flag <= 1'b1;
                                        end else begin
                                            sdram_req_tog   <= ~sdram_req_tog;
                                            sdram_req_op    <= 2'd0;
                                            sdram_req_addr  <= jtag_sdram_ptr;
                                            sdram_req_len   <= 20'd1;
                                            sdram_req_wdata <= {cmd_wdata, stage128[95:0]};
                                            jtag_sdram_ptr  <= jtag_sdram_ptr + 19'd1;
                                        end
                                    end
                                end
                                4'h2: begin
                                    // Explicit WR128 at cmd_addr[18:0] from stage128
                                    if (sdram_cmd_busy) begin
                                        err_flag <= 1'b1;
                                    end else begin
                                        sdram_req_tog   <= ~sdram_req_tog;
                                        sdram_req_op    <= 2'd0;
                                        sdram_req_addr  <= cmd_addr[18:0];
                                        sdram_req_len   <= 20'd1;
                                        sdram_req_wdata <= stage128;
                                    end
                                end
                                4'h3: begin
                                    // RD128 at cmd_addr[18:0]
                                    if (sdram_cmd_busy) begin
                                        err_flag <= 1'b1;
                                    end else begin
                                        sdram_req_tog   <= ~sdram_req_tog;
                                        sdram_req_op    <= 2'd1;
                                        sdram_req_addr  <= cmd_addr[18:0];
                                        sdram_req_len   <= 20'd1;
                                    end
                                end
                                4'h4: begin
                                    // BIST write+verify cmd_wdata[19:0] 128b words from cmd_addr[18:0]
                                    if (sdram_cmd_busy) begin
                                        err_flag <= 1'b1;
                                    end else begin
                                        sdram_req_tog   <= ~sdram_req_tog;
                                        sdram_req_op    <= 2'd2;
                                        sdram_req_addr  <= cmd_addr[18:0];
                                        sdram_req_len   <= cmd_wdata[19:0];
                                    end
                                end
                                default: err_flag <= 1'b1;
                            endcase
                        end else begin
                            err_flag <= 1'b1;
                        end
                    end
                    OP_START: begin
                        if (state == ST_IDLE) begin
                            done_sticky        <= 1'b0;
                            err_flag           <= 1'b0;
                            hw_total_cycles    <= 32'd0;
                            hw_mac_cycles      <= 32'd0;
                            hw_epilogue_cycles <= 32'd0;
                            hw_wgt_reads       <= 32'd0;
                            hw_out_checksum    <= 32'd0;
                            hw_sdram_stall_cyc <= 32'd0;
                            hw_dma_cyc_cnt     <= 32'd0;
                            pass_idx           <= 1'b0;
                            g_t                <= 8'd0;
                            g_t_act_off        <= {ACT_AW{1'b0}};
                            g_j_grp            <= 8'd0;
                            g_j_sub            <= 4'd0;
                            g_j_wgt_off        <= 9'd0;
                            g_kc               <= 8'd0;

                            if (cfg_sdram_stream_en && cfg_op_mode == 2'd0) begin
                                cur_T             <= cfg_dim_T;
                                cur_k16           <= cfg_dim_K[9:4];
                                cur_n16           <= 4'd1;
                                cur_act_base      <= cfg_act_base;
                                cur_wgt_base      <= 9'd0;
                                cur_param_base    <= 7'd0;
                                cur_acc_mode      <= fn_stream_acc_mode(4'd0, cfg_sdram_kchunks, cfg_acc_mode);
                                cur_dst_stride    <= cfg_dst_stride;
                                cur_act_stride    <= cfg_act_stride;
                                cur_dst_is_act    <= cfg_dst_is_act;
                                epi_dst_waddr     <= cfg_dst_is_act ? cfg_hpost_base : cfg_out_base;
                                cur_lut_mode      <= cfg_lut_mode;

                                cur_comp_bank     <= 1'b0;
                                cur_nt            <= 8'd0;
                                cur_kc_tile       <= 4'd0;
                                cur_kc_act_off    <= {ACT_AW{1'b0}};
                                bank0_valid       <= 1'b0;
                                bank1_valid       <= 1'b0;
                                dma_in_flight     <= 1'b0;
                                dma_bank          <= 1'b0;
                                dma_all_issued    <= (cfg_sdram_ntiles == 8'd0 || cfg_sdram_kchunks == 4'd0);
                                dma_nt            <= 8'd0;
                                dma_kc            <= 4'd0;
                                dma_next_addr     <= cfg_sdram_base;
                                dma_words_rx      <= 9'd0;
                                dma_param_pending <= 1'b0;
                                state             <= ST_WAIT_DMA;
                            end else if (cfg_op_mode == 2'd2) begin
                                cur_T          <= cfg_dim_T;
                                cur_k16        <= cfg_dim_K[9:4];
                                cur_act_base   <= cfg_act_base;
                                cur_acc_mode   <= 2'd0;
                                cur_dst_stride <= 8'd1;
                                cur_act_stride <= 8'd0;
                                jn_t           <= 8'd0;
                                jn_t_off       <= {ACT_AW{1'b0}};
                                jn_kc          <= 8'd0;
                                state          <= ST_JOINER_RENC;
                            end else begin
                                cur_T          <= cfg_dim_T;
                                cur_k16        <= cfg_dim_K[9:4];
                                cur_n16        <= cfg_dim_N[7:4];
                                cur_act_base   <= cfg_act_base;
                                cur_wgt_base   <= {1'b0, cfg_wgt_base};
                                cur_param_base <= cfg_param1_base;
                                cur_acc_mode   <= cfg_acc_mode;
                                cur_dst_stride <= cfg_dst_stride;
                                cur_act_stride <= cfg_act_stride;
                                if (cfg_op_mode == 2'd1) begin
                                    epi_dst_waddr  <= cfg_hpost_base;
                                    cur_dst_is_act <= 1'b1;
                                    cur_lut_mode   <= 2'd1;
                                end else begin
                                    if (cfg_dst_is_act) begin
                                        epi_dst_waddr  <= cfg_hpost_base;
                                        cur_dst_is_act <= 1'b1;
                                    end else begin
                                        epi_dst_waddr  <= cfg_out_base;
                                        cur_dst_is_act <= 1'b0;
                                    end
                                    cur_lut_mode <= cfg_lut_mode;
                                end
                                state <= ST_GEMM_RUN;
                            end
                        end else begin
                            err_flag <= 1'b1;
                        end
                    end
                    default: ;
                endcase
            end

            if (state != ST_IDLE) begin
                hw_total_cycles <= hw_total_cycles + 32'd1;
            end
            if (s3_v) begin
                hw_mac_cycles <= hw_mac_cycles + 32'd1;
            end
            if (epi0_v || epi1_v || epi2_v || epi2b_v || epi3_v || epi4_v || epi5_v) begin
                hw_epilogue_cycles <= hw_epilogue_cycles + 32'd1;
            end

            // Background SDRAM DMA receiver & autonomous Ping-Pong prefetcher
            if (sdram_dma_w128_valid) begin
                dma_words_rx <= dma_words_rx + 9'd1;
                if (sdram_dma_w128_idx < 9'd8) begin
                    dma_param_pending <= 1'b1;
                    dma_param_hi_data <= sdram_dma_w128_data[127:64];
                    dma_param_hi_addr <= (dma_bank ? 7'd16 : 7'd0) + {3'd0, sdram_dma_w128_idx[2:0], 1'b1};
                end
            end else if (dma_param_pending) begin
                dma_param_pending <= 1'b0;
            end

            if (dma_in_flight) begin
                hw_dma_cyc_cnt <= hw_dma_cyc_cnt + 32'd1;
                if (!sdram_cmd_busy && (dma_words_rx >= dma_req_words) && !dma_param_pending && !sdram_dma_w128_valid) begin
                    dma_in_flight <= 1'b0;
                    if (!dma_bank) bank0_valid <= 1'b1;
                    else           bank1_valid <= 1'b1;
                    dma_bank <= ~dma_bank;
                end
            end else if (state != ST_IDLE && cfg_sdram_stream_en && !dma_all_issued && !cur_dma_bank_valid && !sdram_cmd_busy) begin
                sdram_req_tog  <= ~sdram_req_tog;
                sdram_req_op   <= 2'd3;
                sdram_req_addr <= dma_next_addr;
                sdram_req_len  <= {11'd0, dma_req_words};
                dma_in_flight  <= 1'b1;
                dma_words_rx   <= 9'd0;
                dma_next_addr  <= dma_next_addr + {10'd0, dma_req_words};
                if ({1'b0, dma_kc} + 5'd1 < {1'b0, cfg_sdram_kchunks}) begin
                    dma_kc <= dma_kc + 4'd1;
                end else begin
                    dma_kc <= 4'd0;
                    if (dma_nt + 8'd1 < cfg_sdram_ntiles) begin
                        dma_nt <= dma_nt + 8'd1;
                    end else begin
                        dma_all_issued <= 1'b1;
                    end
                end
            end

            case (state)
                ST_IDLE: ;
                ST_WAIT_DMA: begin
                    hw_sdram_stall_cyc <= hw_sdram_stall_cyc + 32'd1;
                    if (cur_comp_bank_valid) begin
                        cur_wgt_base   <= {cur_comp_bank, 8'd0};
                        cur_param_base <= cur_comp_bank ? 7'd16 : 7'd0;
                        cur_act_base   <= cfg_act_base + cur_kc_act_off;
                        epi_dst_waddr  <= (cfg_dst_is_act ? cfg_hpost_base : cfg_out_base) + {{(ACT_AW-8){1'b0}}, cur_nt};
                        cur_acc_mode   <= fn_stream_acc_mode(cur_kc_tile, cfg_sdram_kchunks, cfg_acc_mode);
                        g_t            <= 8'd0;
                        g_t_act_off    <= {ACT_AW{1'b0}};
                        g_j_grp        <= 8'd0;
                        g_j_sub        <= 4'd0;
                        g_j_wgt_off    <= 9'd0;
                        g_kc           <= 8'd0;
                        state          <= ST_GEMM_RUN;
                    end
                end
                ST_JOINER_RENC: state <= ST_JOINER_WENC;
                ST_JOINER_WENC: begin
                    jn_enc_buf <= act_rdata128;
                    state      <= ST_JOINER_RDEC;
                end
                ST_JOINER_RDEC: state <= ST_JOINER_WDEC;
                ST_JOINER_WDEC: begin
                    for (si = 0; si < 16; si = si + 1) begin
                        jn_clamped[si] <= join_clamp_comb[si];
                    end
                    jn_lane <= 4'd0;
                    jn_wait <= 2'd0;
                    state   <= ST_JOINER_LUT;
                end
                ST_JOINER_LUT: begin
                    if (jn_wait == 2'd0) begin
                        jn_wait <= 2'd1;
                    end else if (jn_wait == 2'd1) begin
                        jn_wait <= 2'd2;
                    end else if (jn_wait == 2'd2) begin
                        jn_pack[jn_lane] <= lut_rdata;
                        if (jn_lane == 4'd15) begin
                            jn_wait <= 2'd3;
                        end else begin
                            jn_lane <= jn_lane + 4'd1;
                            jn_wait <= 2'd0;
                        end
                    end else begin
                        jn_wait <= 2'd0;
                        if (jn_kc + 8'd1 < cur_k16_chunks) begin
                            jn_kc <= jn_kc + 8'd1;
                            state <= ST_JOINER_RENC;
                        end else begin
                            jn_kc <= 8'd0;
                            if (jn_t + 8'd1 < cur_T) begin
                                jn_t     <= jn_t + 8'd1;
                                jn_t_off <= jn_t_off + {{(ACT_AW-8){1'b0}}, cur_k16_chunks};
                                state    <= ST_JOINER_RENC;
                            end else begin
                                cur_T          <= cfg_dim_T;
                                cur_k16        <= cfg_dim_K[9:4];
                                cur_n16        <= cfg_dim_N[7:4];
                                cur_act_base   <= cfg_hpost_base;
                                cur_wgt_base   <= {1'b0, cfg_wgt_base};
                                cur_param_base <= cfg_param1_base;
                                epi_dst_waddr  <= cfg_out_base;
                                cur_dst_is_act <= 1'b0;
                                cur_lut_mode   <= 2'd0;
                                cur_acc_mode   <= 2'd0;
                                g_t            <= 8'd0;
                                g_t_act_off    <= {ACT_AW{1'b0}};
                                g_j_grp        <= 8'd0;
                                g_j_sub        <= 4'd0;
                                g_j_wgt_off    <= 9'd0;
                                g_kc           <= 8'd0;
                                state          <= ST_GEMM_RUN;
                            end
                        end
                    end
                end

                ST_GEMM_RUN: begin
                    s0_v        <= 1'b1;
                    s0_first_kc <= (g_kc == 8'd0);
                    s0_last_kc  <= (g_kc + 8'd1 == cur_k_chunks);
                    s0_w4_hi    <= g_kc[0];
                    s0_j_sub    <= g_j_sub;
                    s0_elem_idx <= {g_t[(PSUM_AW-5):0], g_j_sub};

                    if (NUM_LANES == 32 || !cfg_w4_mode || !g_kc[0]) begin
                        hw_wgt_reads <= hw_wgt_reads + 32'd1;
                    end

                    if (g_kc + 8'd1 < cur_k_chunks) begin
                        g_kc <= g_kc + 8'd1;
                    end else begin
                        g_kc <= 8'd0;
                        if (g_j_sub != 4'd15) begin
                            g_j_sub     <= g_j_sub + 4'd1;
                            g_j_wgt_off <= g_j_wgt_off + {1'b0, cur_k_chunks};
                        end else begin
                            g_j_sub <= 4'd0;
                            if (g_j_grp + 8'd1 < cur_n16_grps) begin
                                g_j_grp     <= g_j_grp + 8'd1;
                                g_j_wgt_off <= g_j_wgt_off + {1'b0, cur_k_chunks};
                            end else begin
                                g_j_grp     <= 8'd0;
                                g_j_wgt_off <= 9'd0;
                                if (g_t + 8'd1 < cur_T) begin
                                    g_t         <= g_t + 8'd1;
                                    g_t_act_off <= g_t_act_off + {{(ACT_AW-8){1'b0}}, eff_act_stride};
                                end else begin
                                    state <= ST_GEMM_DRAIN;
                                end
                            end
                        end
                    end
                end

                ST_GEMM_DRAIN: begin
                    if (!pipeline_active) begin
                        if (cfg_sdram_stream_en) begin
                            if (!cur_comp_bank) bank0_valid <= 1'b0;
                            else                bank1_valid <= 1'b0;

                            if (stream_last_chunk) begin
                                if (cfg_stress_loop) begin
                                    hw_run_count   <= hw_run_count + 32'd1;
                                    cur_comp_bank  <= 1'b0;
                                    cur_nt         <= 8'd0;
                                    cur_kc_tile    <= 4'd0;
                                    cur_kc_act_off <= {ACT_AW{1'b0}};
                                    bank0_valid    <= 1'b0;
                                    bank1_valid    <= 1'b0;
                                    dma_bank       <= 1'b0;
                                    dma_nt         <= 8'd0;
                                    dma_kc         <= 4'd0;
                                    dma_next_addr  <= cfg_sdram_base;
                                    dma_all_issued <= 1'b0;
                                    state          <= ST_WAIT_DMA;
                                end else begin
                                    done_sticky  <= 1'b1;
                                    hw_run_count <= hw_run_count + 32'd1;
                                    state        <= ST_IDLE;
                                end
                            end else begin
                                cur_comp_bank  <= ~cur_comp_bank;
                                cur_nt         <= next_stream_nt;
                                cur_kc_tile    <= next_stream_kc;
                                cur_kc_act_off <= next_stream_act_off;
                                if (alt_comp_bank_valid) begin
                                    cur_wgt_base   <= {~cur_comp_bank, 8'd0};
                                    cur_param_base <= (~cur_comp_bank) ? 7'd16 : 7'd0;
                                    cur_act_base   <= cfg_act_base + next_stream_act_off;
                                    epi_dst_waddr  <= (cfg_dst_is_act ? cfg_hpost_base : cfg_out_base) + {{(ACT_AW-8){1'b0}}, next_stream_nt};
                                    cur_acc_mode   <= fn_stream_acc_mode(next_stream_kc, cfg_sdram_kchunks, cfg_acc_mode);
                                    g_t            <= 8'd0;
                                    g_t_act_off    <= {ACT_AW{1'b0}};
                                    g_j_grp        <= 8'd0;
                                    g_j_sub        <= 4'd0;
                                    g_j_wgt_off    <= 9'd0;
                                    g_kc           <= 8'd0;
                                    state          <= ST_GEMM_RUN;
                                end else begin
                                    state          <= ST_WAIT_DMA;
                                end
                            end
                        end else if (cfg_op_mode == 2'd1 && !pass_idx) begin
                            pass_idx       <= 1'b1;
                            cur_T          <= cfg_dim_T;
                            cur_k16        <= {2'd0, cfg_dim_N[7:4]};
                            cur_n16        <= cfg_dim_N2[7:4];
                            cur_act_base   <= cfg_hpost_base;
                            cur_wgt_base   <= {1'b0, cfg_wgt2_base};
                            cur_param_base <= cfg_param2_base;
                            epi_dst_waddr  <= cfg_out_base;
                            cur_dst_is_act <= 1'b0;
                            cur_lut_mode   <= 2'd0;
                            cur_acc_mode   <= 2'd0;
                            cur_dst_stride <= 8'd1;
                            cur_act_stride <= 8'd0;
                            g_t            <= 8'd0;
                            g_t_act_off    <= {ACT_AW{1'b0}};
                            g_j_grp        <= 8'd0;
                            g_j_sub        <= 4'd0;
                            g_j_wgt_off    <= 9'd0;
                            g_kc           <= 8'd0;
                            state          <= ST_GEMM_RUN;
                        end else if (cfg_stress_loop) begin
                            hw_run_count <= hw_run_count + 32'd1;
                            g_t          <= 8'd0;
                            g_t_act_off  <= {ACT_AW{1'b0}};
                            g_j_grp      <= 8'd0;
                            g_j_sub      <= 4'd0;
                            g_j_wgt_off  <= 9'd0;
                            g_kc         <= 8'd0;
                            epi_dst_waddr<= cur_dst_is_act ? cfg_hpost_base : cfg_out_base;
                            state        <= ST_GEMM_RUN;
                        end else begin
                            done_sticky  <= 1'b1;
                            hw_run_count <= hw_run_count + 32'd1;
                            state        <= ST_IDLE;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase

            // Stage 0 -> Stage 2
            s2_v        <= s0_v;
            s2_first_kc <= s0_first_kc;
            s2_last_kc  <= s0_last_kc;
            s2_j_sub    <= s0_j_sub;
            s2_elem_idx <= s0_elem_idx;
            s2_bias     <= param_rdata[31:0];
            s2_m16      <= param_rdata[47:32];
            s2_shift    <= param_rdata[52:48];
            for (si = 0; si < NUM_LANES; si = si + 1) begin
                s2_a[si] <= act_rdata_wide[si*8 +: 8];
                s2_w[si] <= s2_w_comb[si];
            end

            // Stage 2 -> Stage 3
            s3_v        <= s2_v;
            s3_first_kc <= s2_first_kc;
            s3_last_kc  <= s2_last_kc;
            s3_j_sub    <= s2_j_sub;
            s3_elem_idx <= s2_elem_idx;
            s3_bias     <= s2_bias;
            s3_m16      <= s2_m16;
            s3_shift    <= s2_shift;
            for (si = 0; si < NUM_LANES; si = si + 1) begin
                s3_prod[si] <= s2_a[si] * s2_w[si];
            end

            // Stage 3 -> Stage 4
            s4_v        <= s3_v;
            s4_first_kc <= s3_first_kc;
            s4_last_kc  <= s3_last_kc;
            s4_j_sub    <= s3_j_sub;
            s4_elem_idx <= s3_elem_idx;
            s4_bias     <= s3_bias;
            s4_m16      <= s3_m16;
            s4_shift    <= s3_shift;
            s4_psum[0]  <= psum_tmp0;
            s4_psum[1]  <= psum_tmp1;
            s4_psum[2]  <= psum_tmp2;
            s4_psum[3]  <= psum_tmp3;

            // Stage 4 -> Stage 5
            s5_v         <= s4_v;
            s5_first_kc  <= s4_first_kc;
            s5_last_kc   <= s4_last_kc;
            s5_j_sub     <= s4_j_sub;
            s5_elem_idx  <= s4_elem_idx;
            s5_bias      <= s4_bias;
            s5_psum_init <= psum_rdata;
            s5_m16       <= s4_m16;
            s5_shift     <= s4_shift;
            s5_sum       <= s4_sum_all;

            // Stage 5 -> Accumulate -> Epilogue 0 (unless store_psum_only)
            if (s5_v) begin
                acc_reg <= acc_next;
                if (s5_last_kc && !store_psum_only) begin
                    epi0_v     <= 1'b1;
                    epi0_acc   <= acc_next;
                    epi0_m16   <= s5_m16;
                    epi0_shift <= s5_shift;
                    epi0_j_sub <= s5_j_sub;
                end
            end

            // Epilogue 0 -> Epilogue 1
            epi1_v     <= epi0_v;
            epi1_prod  <= epi0_acc * $signed({1'b0, epi0_m16});
            epi1_shift <= epi0_shift;
            epi1_j_sub <= epi0_j_sub;

            // Epilogue 1 -> Epilogue 2
            epi2_v       <= epi1_v;
            epi2_rounded <= epi1_prod + round_const;
            epi2_shift   <= epi1_shift;
            epi2_j_sub   <= epi1_j_sub;

            // Epilogue 2 -> Epilogue 2b
            epi2b_v       <= epi2_v;
            epi2b_shifted <= shifted_val;
            epi2b_j_sub   <= epi2_j_sub;

            // Epilogue 2b -> Epilogue 3
            epi3_v     <= epi2b_v;
            epi3_q8    <= clamped_q8;
            epi3_j_sub <= epi2b_j_sub;

            // Epilogue 3 -> Epilogue 4
            epi4_v     <= epi3_v;
            epi4_q8    <= epi3_q8;
            epi4_j_sub <= epi3_j_sub;

            // Epilogue 4 -> Epilogue 5
            if (epi4_v) begin
                out_pack[epi4_j_sub] <= final_byte;
                if (epi4_j_sub == 4'd15) begin
                    epi5_v        <= 1'b1;
                    epi_dst_waddr <= epi_dst_waddr + {1'b0, cur_dst_stride};
                    if (!cur_dst_is_act) begin
                        hw_out_checksum <= {hw_out_checksum[30:0], hw_out_checksum[31]} ^
                                           commit_word128[31:0] ^ commit_word128[63:32] ^
                                           commit_word128[95:64] ^ commit_word128[127:96];
                    end
                end
            end
        end
    end
endmodule
