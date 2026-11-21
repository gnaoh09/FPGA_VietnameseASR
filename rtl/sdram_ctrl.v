`timescale 1ns/1ps

// Controller for Hynix HY57V641620FTP-H 64 Mbit (8 MiB) SDR SDRAM:
//   4 Banks x 4096 Rows x 256 Cols x 16-bit (CL=3, BL=1 pipelined page bursts)
// Supports 4 operations requested from the 80 MHz engine domain via toggle CDC:
//   2'd0 OP_WR128   : Write one 128-bit word (8 x 16-bit) at w128_addr[18:0]
//   2'd1 OP_RD128   : Read  one 128-bit word (8 x 16-bit) at w128_addr[18:0]
//   2'd2 OP_BIST    : Sequential page-burst (256 x 16b/page) write+verify across N words
//   2'd3 OP_DMA_TILE: Sequential page-burst read of N 128-bit words pushed to engine DMA port
module sdram_ctrl #(
    parameter integer CLK_MHZ       = 100,
    parameter integer INIT_WAIT_CYC = 20000 // 200 us @ 100 MHz (reduced in simulation)
) (
    input  wire         clk_sdram,
    input  wire         clk_eng,
    input  wire         rst_n,

    // Engine clock domain interface
    input  wire         eng_req_tog,
    input  wire [1:0]   eng_req_op,
    input  wire [18:0]  eng_addr,
    input  wire [19:0]  eng_len,
    input  wire [127:0] eng_wdata,
    input  wire [2:0]   eng_rd_lat,

    output reg          eng_ack_tog,
    output reg  [127:0] eng_rdata,
    output reg  [31:0]  eng_bist_wr_cyc,
    output reg  [31:0]  eng_bist_rd_cyc,
    output reg  [31:0]  eng_bist_err,
    output reg          eng_sdram_ready,
    output reg          eng_dma_w128_valid,
    output reg  [127:0] eng_dma_w128_data,
    output reg  [8:0]   eng_dma_w128_idx,
    output reg  [63:0]  eng_sdram_cyc_cnt,

    // Physical SDRAM pins (HY57V641620FTP-H TSOP-54)
    output reg          sdram_cke,
    output reg          sdram_cs_n,
    output reg          sdram_ras_n,
    output reg          sdram_cas_n,
    output reg          sdram_we_n,
    output reg  [1:0]   sdram_ba,
    output reg  [11:0]  sdram_addr,
    output reg  [1:0]   sdram_dqm,
    input  wire [15:0]  sdram_dq_in,
    output reg  [15:0]  sdram_dq_out,
    output reg          sdram_dq_oe
);
    // SDRAM Commands {cs_n, ras_n, cas_n, we_n}
    localparam [3:0]
        CMD_NOP   = 4'b0111,
        CMD_ACT   = 4'b0011,
        CMD_READ  = 4'b0101,
        CMD_WRITE = 4'b0100,
        CMD_PRE   = 4'b0010,
        CMD_REF   = 4'b0001,
        CMD_LMR   = 4'b0000;

    // Auto-refresh interval: 64ms / 4096 rows = 15.625 us -> ~14 us safe period
    localparam integer REF_INTERVAL_INT = CLK_MHZ * 14;
    localparam [11:0] REF_INTERVAL = REF_INTERVAL_INT[11:0];
    localparam [3:0]
        T_RP  = 4'd3,  // >= 20 ns
        T_RFC = (CLK_MHZ > 100) ? 4'd9 : 4'd8,  // >= 65 ns
        T_RCD = 4'd3,  // >= 20 ns
        T_WR  = 4'd3;  // >= 15 ns + precharge margin

    // FSM States
    localparam [4:0]
        S_INIT_WAIT  = 5'd0,
        S_INIT_PRE   = 5'd1,
        S_INIT_REF   = 5'd2,
        S_INIT_LMR   = 5'd3,
        S_IDLE       = 5'd4,
        S_REF_PRE    = 5'd5,
        S_REF_CMD    = 5'd6,
        S_ACT        = 5'd7,
        S_WR_BURST   = 5'd8,
        S_WR_RECOV   = 5'd9,
        S_RD_BURST   = 5'd10,
        S_RD_WAIT    = 5'd11,
        S_PRE_DONE   = 5'd12,
        S_NEXT_PAGE  = 5'd13,
        S_DONE       = 5'd14;

    // -------------------------------------------------------------------------
    // CDC: clk_eng -> clk_sdram (Request toggle & latched command parameters)
    // -------------------------------------------------------------------------
    reg req_s0, req_s1, req_s2, req_seen_tog;
    reg        op_is_bist_wr_phase;
    reg [1:0]  lat_op;
    reg [18:0] lat_base_w128;
    reg [19:0] lat_len_w128;
    reg [127:0] lat_wdata;
    reg [2:0]  lat_rd_lat;

    // clk_sdram -> clk_eng CDC signals
    reg         sdram_init_done;
    reg         sdram_ack_tog;
    reg [127:0] sdram_rdata_hold;
    reg [31:0]  sdram_bist_wr_cyc;
    reg [31:0]  sdram_bist_rd_cyc;
    reg [31:0]  sdram_bist_err;

    reg         dma_w128_tog;
    reg [127:0] dma_w128_hold;
    reg [8:0]   dma_w128_idx_hold;

    reg         hb64_tog;
    reg [5:0]   hb64_div;

    // Input I/O register for dq_in (encourages FAST_INPUT_REGISTER)
    reg [15:0] dq_in_d;

    // Burst/page state in clk_sdram domain
    reg [4:0]  state;
    reg [15:0] wait_cnt;
    reg [3:0]  init_ref_cnt;
    reg [11:0] ref_timer;
    reg        ref_due;

    reg [18:0] cur_w128;
    reg [19:0] rem_w128;      // Remaining 128b words for current phase
    reg [22:0] cur_w16_idx;   // Total 16b word offset inside current op
    reg [8:0]  page_w16_rem;  // 16b words to issue before end of 256-word page or rem_w128
    reg [8:0]  dma_out_idx;
    reg        ret_to_burst;  // After refresh, resume page instead of finishing

    // Read pipeline alignment & 128-bit assembly
    reg [6:0]   rd_pipe;
    reg [2:0]   rd_sub_idx;
    reg [111:0] rd_asm128;
    reg [19:0]  bist_rd_w128;
    reg [2:0]   bist_rd_sub;
    reg [31:0]  phase_cyc_cnt;

    // Address decomposition for cur_w128[18:0]:
    //   524,288 128-bit words = 4,194,304 16-bit words (22 bits: w16[21:0])
    //   col[7:0]  = {w128[4:0], sub[2:0]}  (256 x 16b per row = 32 x 128b per row)
    //   ba[1:0]   = w128[6:5]              (4-bank interleave every 32 x 128b words)
    //   row[11:0] = w128[18:7]             (4096 rows)
    wire [11:0] cur_row = cur_w128[18:7];
    wire [1:0]  cur_ba  = cur_w128[6:5];
    wire [7:0]  cur_col = {cur_w128[4:0], cur_w16_idx[2:0]};

    // Deterministic 16-bit BIST pattern (mixes upper row, bank, and sub-word index)
    function [15:0] bist_pat;
        input [18:0] w128;
        input [2:0]  sub;
        begin
            bist_pat = {w128[18:11] ^ w128[7:0], w128[10:6] ^ {2'b01, sub}, sub[2:0] ^ w128[2:0]};
        end
    endfunction

    // Selected write 16-bit word
    wire [15:0] wr_word16 = (lat_op == 2'd0)
        ? lat_wdata[(cur_w16_idx[2:0] * 16) +: 16]
        : bist_pat(cur_w128, cur_w16_idx[2:0]);

    // Expected 16-bit BIST verify word
    wire [15:0] exp_bist_word = bist_pat(bist_rd_w128[18:0], bist_rd_sub);

    wire [31:0] max_w16_to_page_end = 32'd256 - {24'd0, cur_col};
    wire [31:0] rem_w16_total       = {9'd0, rem_w128, 3'b000};
    wire [8:0]  next_burst_len16    = (rem_w16_total < max_w16_to_page_end)
                                      ? rem_w16_total[8:0] : max_w16_to_page_end[8:0];

    always @(posedge clk_sdram or negedge rst_n) begin
        if (!rst_n) begin
            req_s0 <= 1'b0; req_s1 <= 1'b0; req_s2 <= 1'b0; req_seen_tog <= 1'b0;
            sdram_init_done  <= 1'b0;
            sdram_ack_tog    <= 1'b0;
            sdram_rdata_hold <= 128'd0;
            sdram_bist_wr_cyc<= 32'd0;
            sdram_bist_rd_cyc<= 32'd0;
            sdram_bist_err   <= 32'd0;
            dma_w128_tog     <= 1'b0;
            dma_w128_hold    <= 128'd0;
            dma_w128_idx_hold<= 9'd0;
            hb64_tog         <= 1'b0;
            hb64_div         <= 6'd0;

            lat_op           <= 2'd0;
            lat_base_w128    <= 19'd0;
            lat_len_w128     <= 20'd0;
            lat_wdata        <= 128'd0;
            lat_rd_lat       <= 3'd4;
            op_is_bist_wr_phase <= 1'b0;

            sdram_cke   <= 1'b0;
            {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_NOP;
            sdram_ba    <= 2'd0;
            sdram_addr  <= 12'd0;
            sdram_dqm   <= 2'b11;
            sdram_dq_out<= 16'd0;
            sdram_dq_oe <= 1'b0;
            dq_in_d     <= 16'd0;

            state       <= S_INIT_WAIT;
            wait_cnt    <= INIT_WAIT_CYC[15:0];
            init_ref_cnt<= 4'd0;
            ref_timer   <= 12'd0;
            ref_due     <= 1'b0;

            cur_w128    <= 19'd0;
            rem_w128    <= 20'd0;
            cur_w16_idx <= 23'd0;
            page_w16_rem<= 9'd0;
            dma_out_idx <= 9'd0;
            ret_to_burst<= 1'b0;

            rd_pipe     <= 7'd0;
            rd_sub_idx  <= 3'd0;
            rd_asm128   <= 112'd0;
            bist_rd_w128<= 20'd0;
            bist_rd_sub <= 3'd0;
            phase_cyc_cnt <= 32'd0;
        end else begin
            // Register incoming DQ and CDC inputs every clock
            dq_in_d <= sdram_dq_in;
            req_s0  <= eng_req_tog;
            req_s1  <= req_s0;
            req_s2  <= req_s1;

            hb64_div <= hb64_div + 6'd1;
            if (hb64_div == 6'd63) begin
                hb64_tog <= ~hb64_tog;
            end

            // Default SDRAM idle bus signals (NOP, Hi-Z DQ)
            {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_NOP;
            sdram_dq_oe <= 1'b0;
            sdram_dqm   <= sdram_init_done ? 2'b00 : 2'b11;

            // Periodic Auto-Refresh timer (only active after init)
            if (sdram_init_done) begin
                if (ref_timer + 12'd1 >= REF_INTERVAL) begin
                    ref_timer <= 12'd0;
                    ref_due   <= 1'b1;
                end else begin
                    ref_timer <= ref_timer + 12'd1;
                end
            end

            // Read capture pipeline shift & assembly
            rd_pipe <= {rd_pipe[5:0], 1'b0};
            if (rd_pipe[lat_rd_lat]) begin
                if (lat_op == 2'd2 && !op_is_bist_wr_phase) begin
                    // BIST read verify phase
                    if (dq_in_d !== exp_bist_word) begin
                        sdram_bist_err <= sdram_bist_err + 32'd1;
                    end
                    bist_rd_sub <= bist_rd_sub + 3'd1;
                    if (bist_rd_sub == 3'd7) begin
                        bist_rd_w128 <= bist_rd_w128 + 20'd1;
                    end
                end else begin
                    // Assemble 128-bit word from 8 x 16-bit words
                    if (rd_sub_idx != 3'd7) begin
                        rd_asm128[(rd_sub_idx * 16) +: 16] <= dq_in_d;
                    end
                    rd_sub_idx <= rd_sub_idx + 3'd1;
                    if (rd_sub_idx == 3'd7) begin
                        if (lat_op == 2'd3) begin
                            dma_w128_hold     <= {dq_in_d, rd_asm128[111:0]};
                            dma_w128_idx_hold <= dma_out_idx;
                            dma_w128_tog      <= ~dma_w128_tog;
                            dma_out_idx       <= dma_out_idx + 9'd1;
                        end else begin
                            sdram_rdata_hold <= {dq_in_d, rd_asm128[111:0]};
                        end
                    end
                end
            end

            if ((state == S_ACT || state == S_WR_BURST || state == S_WR_RECOV ||
                 state == S_RD_BURST || state == S_RD_WAIT || state == S_PRE_DONE ||
                 state == S_NEXT_PAGE || (state == S_REF_PRE && ret_to_burst) ||
                 (state == S_REF_CMD && ret_to_burst))) begin
                phase_cyc_cnt <= phase_cyc_cnt + 32'd1;
            end

            // Main SDRAM Controller FSM
            if (wait_cnt != 16'd0) begin
                wait_cnt <= wait_cnt - 16'd1;
            end else begin
                case (state)
                    // ---------------------------------------------------------
                    // JEDEC Power-Up Init
                    // ---------------------------------------------------------
                    S_INIT_WAIT: begin
                        sdram_cke <= 1'b1;
                        state     <= S_INIT_PRE;
                    end
                    S_INIT_PRE: begin
                        {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_PRE;
                        sdram_addr   <= 12'b0100_0000_0000; // A10=1 precharge all banks
                        wait_cnt     <= {12'd0, T_RP} - 16'd1;
                        init_ref_cnt <= 4'd8;
                        state        <= S_INIT_REF;
                    end
                    S_INIT_REF: begin
                        if (init_ref_cnt != 4'd0) begin
                            {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_REF;
                            init_ref_cnt <= init_ref_cnt - 4'd1;
                            wait_cnt     <= {12'd0, T_RFC} - 16'd1;
                        end else begin
                            state <= S_INIT_LMR;
                        end
                    end
                    S_INIT_LMR: begin
                        // LMR: Sequential, BL=1 (bits[2:0]=000), CL=3 (bits[6:4]=011), Write Burst=Single
                        {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_LMR;
                        sdram_ba   <= 2'b00;
                        sdram_addr <= 12'b00_1_00_011_0_000;
                        wait_cnt   <= 16'd3;
                        state      <= S_IDLE;
                    end

                    // ---------------------------------------------------------
                    // Idle & Auto-Refresh Service
                    // ---------------------------------------------------------
                    S_IDLE: begin
                        sdram_init_done <= 1'b1;
                        ret_to_burst    <= 1'b0;
                        if (ref_due) begin
                            ref_due <= 1'b0;
                            state   <= S_REF_CMD;
                        end else if (req_s2 != req_seen_tog) begin
                            // New command latched safely across CDC (immune to ref_due stall)
                            req_seen_tog  <= req_s2;
                            lat_op        <= eng_req_op;
                            lat_base_w128 <= eng_addr;
                            lat_len_w128  <= (eng_req_op <= 2'd1) ? 20'd1 : eng_len;
                            lat_wdata     <= eng_wdata;
                            lat_rd_lat    <= (eng_rd_lat < 3'd2) ? 3'd2 : ((eng_rd_lat > 3'd6) ? 3'd6 : eng_rd_lat);

                            cur_w128      <= eng_addr;
                            rem_w128      <= (eng_req_op <= 2'd1) ? 20'd1 : eng_len;
                            cur_w16_idx   <= 23'd0;
                            dma_out_idx   <= 9'd0;
                            rd_sub_idx    <= 3'd0;
                            bist_rd_w128  <= {1'b0, eng_addr};
                            bist_rd_sub   <= 3'd0;
                            phase_cyc_cnt <= 32'd0;

                            if (eng_req_op == 2'd2) begin
                                sdram_bist_err    <= 32'd0;
                                sdram_bist_wr_cyc <= 32'd0;
                                sdram_bist_rd_cyc <= 32'd0;
                                op_is_bist_wr_phase <= 1'b1;
                            end else begin
                                op_is_bist_wr_phase <= 1'b0;
                            end
                            state <= S_ACT;
                        end
                    end

                    S_REF_PRE: begin
                        {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_PRE;
                        sdram_addr <= 12'b0100_0000_0000;
                        wait_cnt   <= {12'd0, T_RP} - 16'd1;
                        state      <= S_REF_CMD;
                    end

                    S_REF_CMD: begin
                        {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_REF;
                        wait_cnt <= {12'd0, T_RFC} - 16'd1;
                        state    <= ret_to_burst ? S_ACT : S_IDLE;
                    end

                    // ---------------------------------------------------------
                    // Activate Page & Stream Consecutive Page-Burst Commands
                    // ---------------------------------------------------------
                    S_ACT: begin
                        {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_ACT;
                        sdram_ba     <= cur_ba;
                        sdram_addr   <= cur_row;
                        page_w16_rem <= next_burst_len16;
                        wait_cnt     <= {12'd0, T_RCD} - 16'd1;
                        if (lat_op == 2'd0 || (lat_op == 2'd2 && op_is_bist_wr_phase)) begin
                            state <= S_WR_BURST;
                        end else begin
                            state <= S_RD_BURST;
                        end
                    end

                    S_WR_BURST: begin
                        {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_WRITE;
                        sdram_ba     <= cur_ba;
                        sdram_addr   <= {4'b0000, cur_col}; // A10=0: no auto-precharge
                        sdram_dq_out <= wr_word16;
                        sdram_dq_oe  <= 1'b1;
                        sdram_dqm    <= 2'b00;

                        cur_w16_idx  <= cur_w16_idx + 23'd1;
                        if (cur_w16_idx[2:0] == 3'd7) begin
                            cur_w128 <= cur_w128 + 19'd1;
                            rem_w128 <= rem_w128 - 20'd1;
                        end

                        page_w16_rem <= page_w16_rem - 9'd1;
                        if (page_w16_rem == 9'd1) begin
                            wait_cnt <= {12'd0, T_WR} - 16'd1;
                            state    <= S_WR_RECOV;
                        end
                    end

                    S_WR_RECOV: begin
                        {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_PRE;
                        sdram_addr <= 12'b0100_0000_0000;
                        wait_cnt   <= {12'd0, T_RP} - 16'd1;
                        state      <= S_NEXT_PAGE;
                    end

                    S_RD_BURST: begin
                        {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_READ;
                        sdram_ba    <= cur_ba;
                        sdram_addr  <= {4'b0000, cur_col}; // A10=0: manual precharge after page
                        sdram_dqm   <= 2'b00;
                        rd_pipe[0]  <= 1'b1;

                        cur_w16_idx <= cur_w16_idx + 23'd1;
                        if (cur_w16_idx[2:0] == 3'd7) begin
                            cur_w128 <= cur_w128 + 19'd1;
                            rem_w128 <= rem_w128 - 20'd1;
                        end

                        page_w16_rem <= page_w16_rem - 9'd1;
                        if (page_w16_rem == 9'd1) begin
                            // Wait for in-flight reads to capture into dq_in_d & process
                            wait_cnt <= {13'd0, lat_rd_lat} + 16'd2;
                            state    <= S_RD_WAIT;
                        end
                    end

                    S_RD_WAIT: begin
                        {sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n} <= CMD_PRE;
                        sdram_addr <= 12'b0100_0000_0000;
                        wait_cnt   <= {12'd0, T_RP} - 16'd1;
                        state      <= S_PRE_DONE;
                    end

                    S_PRE_DONE: begin
                        state <= S_NEXT_PAGE;
                    end

                    S_NEXT_PAGE: begin
                        if (rem_w128 != 20'd0) begin
                            if (ref_due) begin
                                ref_due      <= 1'b0;
                                ret_to_burst <= 1'b1;
                                state        <= S_REF_CMD; // Already precharged!
                            end else begin
                                state        <= S_ACT;
                            end
                        end else begin
                            if (lat_op == 2'd2 && op_is_bist_wr_phase) begin
                                // Transition from BIST Write phase -> BIST Read Verify phase
                                sdram_bist_wr_cyc   <= phase_cyc_cnt;
                                phase_cyc_cnt       <= 32'd0;
                                op_is_bist_wr_phase <= 1'b0;
                                cur_w128            <= lat_base_w128;
                                rem_w128            <= lat_len_w128;
                                cur_w16_idx         <= 23'd0;
                                bist_rd_w128        <= {1'b0, lat_base_w128};
                                bist_rd_sub         <= 3'd0;
                                if (ref_due) begin
                                    ref_due      <= 1'b0;
                                    ret_to_burst <= 1'b1;
                                    state        <= S_REF_CMD;
                                end else begin
                                    state        <= S_ACT;
                                end
                            end else begin
                                if (lat_op == 2'd2) begin
                                    sdram_bist_rd_cyc <= phase_cyc_cnt;
                                end
                                state <= S_DONE;
                            end
                        end
                    end

                    S_DONE: begin
                        sdram_ack_tog <= req_seen_tog;
                        ret_to_burst  <= 1'b0;
                        state         <= S_IDLE;
                    end

                    default: state <= S_IDLE;
                endcase
            end
        end
    end

    // -------------------------------------------------------------------------
    // CDC: clk_sdram -> clk_eng Synchronizers
    // -------------------------------------------------------------------------
    reg init_s0, init_s1;
    reg ack_s0, ack_s1;
    reg dma_s0, dma_s1, dma_s2;
    reg hb_s0, hb_s1, hb_s2;

    always @(posedge clk_eng or negedge rst_n) begin
        if (!rst_n) begin
            init_s0 <= 1'b0; init_s1 <= 1'b0; eng_sdram_ready <= 1'b0;
            ack_s0  <= 1'b0; ack_s1  <= 1'b0; eng_ack_tog     <= 1'b0;
            dma_s0  <= 1'b0; dma_s1  <= 1'b0; dma_s2          <= 1'b0;
            hb_s0   <= 1'b0; hb_s1   <= 1'b0; hb_s2           <= 1'b0;
            eng_rdata         <= 128'd0;
            eng_bist_wr_cyc   <= 32'd0;
            eng_bist_rd_cyc   <= 32'd0;
            eng_bist_err      <= 32'd0;
            eng_dma_w128_valid<= 1'b0;
            eng_dma_w128_data <= 128'd0;
            eng_dma_w128_idx  <= 9'd0;
            eng_sdram_cyc_cnt <= 64'd0;
        end else begin
            init_s0 <= sdram_init_done;
            init_s1 <= init_s0;
            eng_sdram_ready <= init_s1;

            ack_s0 <= sdram_ack_tog;
            ack_s1 <= ack_s0;
            if (ack_s1 != eng_ack_tog) begin
                eng_ack_tog     <= ack_s1;
                eng_rdata       <= sdram_rdata_hold;
                eng_bist_wr_cyc <= sdram_bist_wr_cyc;
                eng_bist_rd_cyc <= sdram_bist_rd_cyc;
                eng_bist_err    <= sdram_bist_err;
            end

            dma_s0 <= dma_w128_tog;
            dma_s1 <= dma_s0;
            dma_s2 <= dma_s1;
            eng_dma_w128_valid <= 1'b0;
            if (dma_s1 != dma_s2) begin
                eng_dma_w128_valid <= 1'b1;
                eng_dma_w128_data  <= dma_w128_hold;
                eng_dma_w128_idx   <= dma_w128_idx_hold;
            end

            hb_s0 <= hb64_tog;
            hb_s1 <= hb_s0;
            hb_s2 <= hb_s1;
            if (hb_s1 != hb_s2) begin
                eng_sdram_cyc_cnt <= eng_sdram_cyc_cnt + 64'd64;
            end
        end
    end
endmodule
