`timescale 1ns/1ps

module tb_top;
    parameter integer TB_NUM_LANES = 16;

    reg        clk   = 1'b0;
    reg        rst_n = 1'b0;
    reg  [3:0] key   = 4'b1111;
    wire [3:0] led;

    reg        sim_tck = 1'b0;
    reg        sim_tms = 1'b0;
    reg        sim_tdi = 1'b0;
    wire       sim_tdo;

    wire        sdram_clk;
    wire        sdram_cke;
    wire        sdram_cs_n;
    wire        sdram_ras_n;
    wire        sdram_cas_n;
    wire        sdram_we_n;
    wire [1:0]  sdram_ba;
    wire [11:0] sdram_addr;
    wire [1:0]  sdram_dqm;
    wire [15:0] sdram_dq;

    // Behavioral HY57V641620 SDR SDRAM Model (CL=3, BL=1, 64K x 16b test window)
    reg [15:0] sdram_mem [0:65535];
    reg [11:0] sdram_act_row [0:3];
    reg        sdram_rd_v0, sdram_rd_v1;
    reg [15:0] sdram_rd_d0, sdram_rd_d1;
    reg [15:0] sdram_dq_drv;
    reg        sdram_dq_en;
    assign sdram_dq = sdram_dq_en ? sdram_dq_drv : 16'hZZZZ;

    integer m_i;
    initial begin
        for (m_i = 0; m_i < 65536; m_i = m_i + 1) sdram_mem[m_i] = 16'd0;
        for (m_i = 0; m_i < 4; m_i = m_i + 1) sdram_act_row[m_i] = 12'd0;
        sdram_rd_v0  = 1'b0; sdram_rd_v1  = 1'b0;
        sdram_rd_d0  = 16'd0; sdram_rd_d1 = 16'd0;
        sdram_dq_drv = 16'd0; sdram_dq_en = 1'b0;
    end

    always @(posedge sdram_clk) begin
        sdram_dq_en <= 1'b0;
        sdram_rd_v1 <= sdram_rd_v0;
        sdram_rd_d1 <= sdram_rd_d0;
        if (sdram_rd_v1) begin
            sdram_dq_en  <= 1'b1;
            sdram_dq_drv <= sdram_rd_d1;
        end
        sdram_rd_v0 <= 1'b0;
        if (sdram_cke && !sdram_cs_n) begin
            case ({sdram_ras_n, sdram_cas_n, sdram_we_n})
                3'b011: sdram_act_row[sdram_ba] <= sdram_addr; // ACTIVE
                3'b100: begin
                    if (sdram_dqm == 2'b00) begin
                        sdram_mem[{sdram_act_row[sdram_ba][5:0], sdram_ba, sdram_addr[7:0]}] <= sdram_dq;
                    end
                end
                3'b101: begin
                    sdram_rd_v0 <= 1'b1;
                    sdram_rd_d0 <= sdram_mem[{sdram_act_row[sdram_ba][5:0], sdram_ba, sdram_addr[7:0]}];
                end
                default: ;
            endcase
        end
    end

    top #(.NUM_LANES(TB_NUM_LANES)) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .key         (key),
        .led         (led),
        .sdram_clk   (sdram_clk),
        .sdram_cke   (sdram_cke),
        .sdram_cs_n  (sdram_cs_n),
        .sdram_ras_n (sdram_ras_n),
        .sdram_cas_n (sdram_cas_n),
        .sdram_we_n  (sdram_we_n),
        .sdram_ba    (sdram_ba),
        .sdram_addr  (sdram_addr),
        .sdram_dqm   (sdram_dqm),
        .sdram_dq    (sdram_dq),
        .sim_tck     (sim_tck),
        .sim_tms     (sim_tms),
        .sim_tdi     (sim_tdi),
        .sim_tdo     (sim_tdo)
    );

    // 50 MHz system clock (20 ns period)
    always #10 clk = ~clk;

    // JTAG Opcodes & IRs
    localparam [9:0] IR_USER0 = 10'h00C;
    localparam [9:0] IR_USER1 = 10'h00E;

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

    // -------------------------------------------------------------------------
    // Bitbang JTAG Driver Tasks (5 MHz JTAG TCK << 50 MHz clk)
    // -------------------------------------------------------------------------
    task jtag_pulse(input reg tms_val, input reg tdi_val, output reg tdo_val);
        begin
            sim_tms = tms_val;
            sim_tdi = tdi_val;
            #25;
            tdo_val = sim_tdo; // Sample TDO before rising edge (matches USB-Blaster READ)
            sim_tck = 1'b1;
            #30;
            sim_tck = 1'b0;
            #5;
        end
    endtask

    task jtag_reset_to_rti;
        reg dummy;
        integer k;
        begin
            for (k = 0; k < 5; k = k + 1) jtag_pulse(1'b1, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy); // RTI
        end
    endtask

    task jtag_ir_scan(input [9:0] ir_val);
        reg dummy;
        integer k;
        begin
            // RTI -> Select-DR -> Select-IR -> Capture-IR -> Shift-IR
            jtag_pulse(1'b1, 1'b0, dummy);
            jtag_pulse(1'b1, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy);
            for (k = 0; k < 10; k = k + 1) begin
                jtag_pulse((k == 9) ? 1'b1 : 1'b0, ir_val[k], dummy);
            end
            // Exit1-IR -> Update-IR -> RTI
            jtag_pulse(1'b1, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy);
            // Allow 50 MHz domain idle ticks
            repeat (4) @(posedge clk);
        end
    endtask

    reg [9:0] cur_ir = 10'h000;

    task jtag_send_cmd(input [7:0] op, input [23:0] addr, input [31:0] wdata);
        reg [63:0] frame;
        reg dummy;
        integer k;
        begin
            if (cur_ir != IR_USER0) begin
                jtag_ir_scan(IR_USER0);
                cur_ir = IR_USER0;
            end
            frame = {op, addr, wdata};
            // RTI -> Select-DR -> Capture-DR -> Shift-DR
            jtag_pulse(1'b1, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy);
            for (k = 0; k < 64; k = k + 1) begin
                jtag_pulse((k == 63) ? 1'b1 : 1'b0, frame[k], dummy);
            end
            // Exit1-DR -> Update-DR -> RTI
            jtag_pulse(1'b1, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy);
            // Wait for 3-FF synchronizer + 1-cycle BRAM commit
            repeat (8) @(posedge clk);
        end
    endtask

    task jtag_read_resp(input [15:0] sel, output [63:0] rdata);
        reg bit_out;
        reg dummy;
        integer k;
        begin
            jtag_send_cmd(OP_SET_READ_PTR, {8'h00, sel}, 32'd0);
            if (cur_ir != IR_USER1) begin
                jtag_ir_scan(IR_USER1);
                cur_ir = IR_USER1;
            end
            // RTI -> Select-DR -> Capture-DR -> Shift-DR
            jtag_pulse(1'b1, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy);
            rdata = 64'd0;
            for (k = 0; k < 64; k = k + 1) begin
                jtag_pulse((k == 63) ? 1'b1 : 1'b0, 1'b0, bit_out);
                rdata[k] = bit_out;
            end
            // Exit1-DR -> Update-DR -> RTI
            jtag_pulse(1'b1, 1'b0, dummy);
            jtag_pulse(1'b0, 1'b0, dummy);
        end
    endtask

    task jtag_write_128b(input [7:0] commit_op, input [7:0] word_addr, input [127:0] word);
        begin
            jtag_send_cmd(OP_STAGE_W32, 24'd0, word[31:0]);
            jtag_send_cmd(OP_STAGE_W32, 24'd1, word[63:32]);
            jtag_send_cmd(OP_STAGE_W32, 24'd2, word[95:64]);
            jtag_send_cmd(OP_STAGE_W32, 24'd3, word[127:96]);
            jtag_send_cmd(commit_op, {16'd0, word_addr}, 32'd0);
        end
    endtask

    task jtag_write_64b_param(input [7:0] p_addr, input [63:0] word);
        begin
            jtag_send_cmd(OP_STAGE_W32, 24'd0, word[31:0]);
            jtag_send_cmd(OP_STAGE_W32, 24'd1, word[63:32]);
            jtag_send_cmd(OP_WRITE_PARAM, {16'd0, p_addr}, 32'd0);
        end
    endtask

    task jtag_read_128b(input [3:0] mem_id, input [7:0] word_addr, output [127:0] word);
        reg [63:0] lo, hi;
        begin
            jtag_read_resp({mem_id, 3'b000, 1'b0, word_addr}, lo);
            jtag_read_resp({mem_id, 3'b000, 1'b1, word_addr}, hi);
            word = {hi, lo};
        end
    endtask

    // Memories for Golden Vector Loading
    reg [127:0] gv_act   [0:31];
    reg [127:0] gv_dec   [0:1];
    reg [127:0] gv_wgt   [0:255];
    reg  [63:0] gv_param [0:95];
    reg   [7:0] gv_lut   [0:255];
    reg [127:0] gv_hpost [0:63];
    reg [127:0] gv_yout  [0:31];

    reg [63:0]  resp64;
    reg [127:0] got128;
    integer i, err_cnt, poll_cnt;
    reg [31:0] w8_wgt_reads, w4_wgt_reads;

    task wait_done(output [31:0] tot_cyc, output [31:0] mac_cyc, output [31:0] epi_cyc, output [31:0] w_reads, output [31:0] csum);
        reg [63:0] st, c1, c2, c3;
        begin
            poll_cnt = 0;
            st = 64'd0;
            while (st[1] == 1'b0 && poll_cnt < 50) begin
                jtag_read_resp(16'h2000, st);
                poll_cnt = poll_cnt + 1;
            end
            if (st[1] !== 1'b1) begin
                $display("FAIL: Engine timed out waiting for done! st=%h state=%0d pipe=%b tot=%0d pass=%0d g_t=%0d g_j_grp=%0d g_j_sub=%0d g_kc=%0d",
                         st, dut.u_engine.state, dut.u_engine.pipeline_active, dut.u_engine.hw_total_cycles,
                         dut.u_engine.pass_idx, dut.u_engine.g_t, dut.u_engine.g_j_grp, dut.u_engine.g_j_sub, dut.u_engine.g_kc);
                $finish(1);
            end
            jtag_read_resp(16'h2001, c1);
            jtag_read_resp(16'h2002, c2);
            jtag_read_resp(16'h2003, c3);
            tot_cyc = c1[31:0];
            mac_cyc = c1[63:32];
            epi_cyc = c2[31:0];
            w_reads = c2[63:32];
            csum    = c3[31:0];
        end
    endtask

    reg [31:0] tot_c, mac_c, epi_c, wr_c, csum_c;

    initial begin
        #50;
        rst_n = 1'b1;
        repeat (5) @(posedge clk);
        jtag_reset_to_rti;

        // 0. Check MAGIC & Capabilities
        jtag_read_resp(16'h2000, resp64);
        if (resp64[63:32] !== 32'h5A463445) begin
            $display("FAIL: MAGIC mismatch, got %h", resp64[63:32]);
            $finish(1);
        end
        jtag_read_resp(16'h2005, resp64);
        $display("[TB] Connected MAGIC=0x5A463445, NUM_LANES=%0d, VER=0x%h", resp64[31:16], resp64[15:0]);

        // ---------------------------------------------------------------------
        // TEST 1: stage0_layer0_ffn1_W8A8 (Mode S: L1 + SwooshL -> L2)
        // ---------------------------------------------------------------------
        $readmemh("golden_vectors/stage0_layer0_ffn1_W8A8/act_words_128b.hex", gv_act);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W8A8/wgt_words_128b.hex", gv_wgt);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W8A8/param_words_64b.hex", gv_param);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W8A8/swoosh_l_lut.hex", gv_lut);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W8A8/h_post_words_128b.hex", gv_hpost);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W8A8/y_golden_words_128b.hex", gv_yout);

        for (i = 0; i < 32; i = i + 1)  jtag_write_128b(OP_COMMIT_ACT, i[7:0], gv_act[i]);
        for (i = 0; i < 256; i = i + 1) jtag_write_128b(OP_COMMIT_WGT, i[7:0], gv_wgt[i]);
        for (i = 0; i < 96; i = i + 1)  jtag_write_64b_param(i[7:0], gv_param[i]);
        for (i = 0; i < 256; i = i + 1) jtag_send_cmd(OP_WRITE_LUT, {15'd0, i[8:0]}, {24'd0, gv_lut[i]});

        // reg_cfg0: op_mode=1, w4=0, lut=1, T=16, K=32, N=64
        jtag_send_cmd(OP_WRITE_REG, 24'd0, {3'd0, 8'd64, 8'd32, 8'd16, 2'd1, 1'b0, 2'd1});
        // reg_cfg1: N2=32, act_base=0, wgt_base=0, hpost_base=64
        jtag_send_cmd(OP_WRITE_REG, 24'd1, {8'd64, 8'd0, 8'd0, 8'd32});
        // reg_cfg2: wgt2_base=128, param1_base=0, param2_base=64, out_base=0
        jtag_send_cmd(OP_WRITE_REG, 24'd2, {8'd0, 8'd64, 8'd0, 8'd128});
        jtag_send_cmd(OP_WRITE_REG, 24'd3, 32'd0);
        jtag_send_cmd(OP_START, 24'd0, 32'd0);

        wait_done(tot_c, mac_c, epi_c, w8_wgt_reads, csum_c);
        err_cnt = 0;
        for (i = 0; i < 64; i = i + 1) begin
            jtag_read_128b(4'h1, 8'd64 + i[7:0], got128);
            if (got128 !== gv_hpost[i]) begin
                $display("FAIL FFN_W8A8 h_post[%0d]: got=%h exp=%h", i, got128, gv_hpost[i]);
                err_cnt = err_cnt + 1;
            end
        end
        for (i = 0; i < 32; i = i + 1) begin
            jtag_read_128b(4'h0, i[7:0], got128);
            if (got128 !== gv_yout[i]) begin
                $display("FAIL FFN_W8A8 y_out[%0d]: got=%h exp=%h", i, got128, gv_yout[i]);
                err_cnt = err_cnt + 1;
            end
        end
        if (err_cnt != 0) $finish(1);
        $display("PASS Test 1 (stage0_layer0_ffn1_W8A8): 0 LSB mismatch! total_cyc=%0d mac_cyc=%0d epi_cyc=%0d wgt_reads=%0d csum=0x%08x",
                 tot_c, mac_c, epi_c, w8_wgt_reads, csum_c);

        // ---------------------------------------------------------------------
        // TEST 2: stage0_layer0_ffn1_W4A8 (Mode S: Hardware W4 Unpack + L1+SwooshL+L2)
        // ---------------------------------------------------------------------
        $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/act_words_128b.hex", gv_act);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/wgt_words_128b.hex", gv_wgt);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/param_words_64b.hex", gv_param);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/swoosh_l_lut.hex", gv_lut);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/h_post_words_128b.hex", gv_hpost);
        $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/y_golden_words_128b.hex", gv_yout);

        for (i = 0; i < 32; i = i + 1)  jtag_write_128b(OP_COMMIT_ACT, i[7:0], gv_act[i]);
        for (i = 0; i < 128; i = i + 1) jtag_write_128b(OP_COMMIT_WGT, i[7:0], gv_wgt[i]);
        for (i = 0; i < 96; i = i + 1)  jtag_write_64b_param(i[7:0], gv_param[i]);
        for (i = 0; i < 256; i = i + 1) jtag_send_cmd(OP_WRITE_LUT, {15'd0, i[8:0]}, {24'd0, gv_lut[i]});

        // reg_cfg0: op_mode=1, w4=1, lut=1, T=16, K=32, N=64
        jtag_send_cmd(OP_WRITE_REG, 24'd0, {3'd0, 8'd64, 8'd32, 8'd16, 2'd1, 1'b1, 2'd1});
        jtag_send_cmd(OP_WRITE_REG, 24'd1, {8'd64, 8'd0, 8'd0, 8'd32});
        // wgt2_base=64 in W4 mode!
        jtag_send_cmd(OP_WRITE_REG, 24'd2, {8'd0, 8'd64, 8'd0, 8'd64});
        jtag_send_cmd(OP_START, 24'd0, 32'd0);

        wait_done(tot_c, mac_c, epi_c, w4_wgt_reads, csum_c);
        err_cnt = 0;
        for (i = 0; i < 64; i = i + 1) begin
            jtag_read_128b(4'h1, 8'd64 + i[7:0], got128);
            if (got128 !== gv_hpost[i]) begin
                $display("FAIL FFN_W4A8 h_post[%0d]: got=%h exp=%h", i, got128, gv_hpost[i]);
                err_cnt = err_cnt + 1;
            end
        end
        for (i = 0; i < 32; i = i + 1) begin
            jtag_read_128b(4'h0, i[7:0], got128);
            if (got128 !== gv_yout[i]) begin
                $display("FAIL FFN_W4A8 y_out[%0d]: got=%h exp=%h", i, got128, gv_yout[i]);
                err_cnt = err_cnt + 1;
            end
        end
        if (err_cnt != 0) $finish(1);
        $display("PASS Test 2 (stage0_layer0_ffn1_W4A8): 0 LSB mismatch! total_cyc=%0d mac_cyc=%0d epi_cyc=%0d wgt_reads=%0d (50%% of W8) csum=0x%08x",
                 tot_c, mac_c, epi_c, w4_wgt_reads, csum_c);

        // ---------------------------------------------------------------------
        // TEST 3: joiner_W8A8 (Mode S: Add + Tanh LUT + Linear)
        // ---------------------------------------------------------------------
        $readmemh("golden_vectors/joiner_W8A8/enc_words_128b.hex", gv_act);
        $readmemh("golden_vectors/joiner_W8A8/dec_words_128b.hex", gv_dec);
        $readmemh("golden_vectors/joiner_W8A8/wgt_words_128b.hex", gv_wgt);
        $readmemh("golden_vectors/joiner_W8A8/param_words_64b.hex", gv_param);
        $readmemh("golden_vectors/joiner_W8A8/tanh_lut.hex", gv_lut);
        $readmemh("golden_vectors/joiner_W8A8/logit_golden_words_128b.hex", gv_yout);

        for (i = 0; i < 32; i = i + 1) jtag_write_128b(OP_COMMIT_ACT, i[7:0], gv_act[i]);
        for (i = 0; i < 2; i = i + 1)  jtag_write_128b(OP_COMMIT_ACT, 8'd32 + i[7:0], gv_dec[i]);
        for (i = 0; i < 64; i = i + 1) jtag_write_128b(OP_COMMIT_WGT, i[7:0], gv_wgt[i]);
        for (i = 0; i < 32; i = i + 1) jtag_write_64b_param(i[7:0], gv_param[i]);
        // Load Tanh LUT into bank 1 (addr 256..511)
        for (i = 0; i < 256; i = i + 1) jtag_send_cmd(OP_WRITE_LUT, {14'd0, 1'b1, i[7:0]}, {24'd0, gv_lut[i]});

        // reg_cfg0: op_mode=2 (Joiner), w4=0, lut=2, T=16, D=32, V=32
        jtag_send_cmd(OP_WRITE_REG, 24'd0, {3'd0, 8'd32, 8'd32, 8'd16, 2'd2, 1'b0, 2'd2});
        jtag_send_cmd(OP_WRITE_REG, 24'd1, {8'd64, 8'd0, 8'd0, 8'd0});
        jtag_send_cmd(OP_WRITE_REG, 24'd2, {8'd0, 8'd0, 8'd0, 8'd0});
        jtag_send_cmd(OP_WRITE_REG, 24'd3, 32'd32); // dec_base=32
        jtag_send_cmd(OP_START, 24'd0, 32'd0);

        wait_done(tot_c, mac_c, epi_c, wr_c, csum_c);
        err_cnt = 0;
        for (i = 0; i < 32; i = i + 1) begin
            jtag_read_128b(4'h0, i[7:0], got128);
            if (got128 !== gv_yout[i]) begin
                $display("FAIL joiner_W8A8 logit_out[%0d]: got=%h exp=%h", i, got128, gv_yout[i]);
                err_cnt = err_cnt + 1;
            end
        end
        if (err_cnt != 0) $finish(1);
        $display("PASS Test 3 (joiner_W8A8): 0 LSB mismatch! total_cyc=%0d mac_cyc=%0d epi_cyc=%0d wgt_reads=%0d csum=0x%08x",
                 tot_c, mac_c, epi_c, wr_c, csum_c);

        // ---------------------------------------------------------------------
        // TEST 4: SDRAM Controller (WR128/RD128/PUSH_W32_AUTO + Page-Burst BIST)
        // ---------------------------------------------------------------------
        begin : test4_sdram_bist
            reg [63:0] st_sd, cyc0, cyc1, bist_c, bist_e;
            reg [127:0] test_w, read_w;
            integer t4_wait, k_i;
            t4_wait = 0;
            st_sd = 64'd0;
            while (!st_sd[32] && t4_wait < 50) begin
                jtag_read_resp(16'h2005, st_sd);
                t4_wait = t4_wait + 1;
            end
            if (!st_sd[32]) begin
                $display("FAIL Test 4: sdram_ready never asserted (0x2005=%h)", st_sd);
                $finish(1);
            end
            jtag_read_resp(16'h200A, cyc0);
            jtag_send_cmd(OP_SDRAM_CMD, 24'h000000, 32'd0); // SET_ADDR = 0
            for (k_i = 0; k_i < 4; k_i = k_i + 1) begin
                test_w = {32'hCAFE0000 + k_i, 32'hBEEF1000 + k_i, 32'h12340000 + k_i, 32'hA5A50000 + k_i};
                jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, test_w[31:0]);
                jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, test_w[63:32]);
                jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, test_w[95:64]);
                jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, test_w[127:96]);
            end
            for (k_i = 0; k_i < 4; k_i = k_i + 1) begin
                test_w = {32'hCAFE0000 + k_i, 32'hBEEF1000 + k_i, 32'h12340000 + k_i, 32'hA5A50000 + k_i};
                jtag_send_cmd(OP_SDRAM_CMD, 24'h300000 | k_i[18:0], 32'd0); // RD128
                t4_wait = 0;
                st_sd = {64{1'b1}};
                while (st_sd[36] && t4_wait < 20) begin
                    jtag_read_resp(16'h2005, st_sd);
                    t4_wait = t4_wait + 1;
                end
                jtag_read_resp(16'h2006, read_w[63:0]);
                jtag_read_resp(16'h2007, read_w[127:64]);
                if (read_w !== test_w) begin
                    $display("FAIL Test 4 RD128[%0d]: got=%h exp=%h", k_i, read_w, test_w);
                    $finish(1);
                end
            end
            // Page-burst BIST across 256 words (4 KiB, crosses 8 rows/4 banks!)
            jtag_send_cmd(OP_SDRAM_CMD, 24'h400000, 32'd256);
            t4_wait = 0;
            st_sd = {64{1'b1}};
            while (st_sd[36] && t4_wait < 200) begin
                jtag_read_resp(16'h2005, st_sd);
                t4_wait = t4_wait + 1;
            end
            jtag_read_resp(16'h2008, bist_c);
            jtag_read_resp(16'h2009, bist_e);
            jtag_read_resp(16'h200A, cyc1);
            if (st_sd[36] || bist_e[31:0] !== 32'd0) begin
                $display("FAIL Test 4 BIST: busy=%0d err=%0d wr_cyc=%0d rd_cyc=%0d",
                         st_sd[36], bist_e[31:0], bist_c[31:0], bist_c[63:32]);
                $finish(1);
            end
            $display("PASS Test 4 (SDRAM WR/RD128 + 256w BIST): 0 errors! wr_cyc=%0d rd_cyc=%0d sdram_cyc_delta=%0d",
                     bist_c[31:0], bist_c[63:32], cyc1 - cyc0);
        end

        // ---------------------------------------------------------------------
        // TEST 5: Multi-Tile Ping-Pong SDRAM DMA End-to-End (FFN_W4A8 + K-Chunk)
        // ---------------------------------------------------------------------
        begin : test5_sdram_dma
            integer nt_i, wi_i, kc_i, k16_i, row_i, t_i;
            reg [63:0] stall_info, dma_info;
            reg [127:0] pw_word, act_w, exp_w;
            reg [3:0] w4_nib;
            integer signed mac_acc;

            // Test 5a: Pack stage0_layer0_ffn1_W4A8 L1 (4 N-tiles, K=32, 1 chunk, 8+16=24w/tile) at SDRAM=0x100
            $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/act_words_128b.hex", gv_act);
            $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/wgt_words_128b.hex", gv_wgt);
            $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/param_words_64b.hex", gv_param);
            $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/swoosh_l_lut.hex", gv_lut);
            $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/h_post_words_128b.hex", gv_hpost);
            $readmemh("golden_vectors/stage0_layer0_ffn1_W4A8/y_golden_words_128b.hex", gv_yout);

            for (i = 0; i < 32; i = i + 1)  jtag_write_128b(OP_COMMIT_ACT, i[7:0], gv_act[i]);
            for (i = 0; i < 256; i = i + 1) jtag_send_cmd(OP_WRITE_LUT, {15'd0, i[8:0]}, {24'd0, gv_lut[i]});

            jtag_send_cmd(OP_SDRAM_CMD, 24'h000000, 32'd256); // SET_ADDR = 256
            for (nt_i = 0; nt_i < 4; nt_i = nt_i + 1) begin
                for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1) begin
                    pw_word = {gv_param[nt_i*16 + wi_i*2 + 1], gv_param[nt_i*16 + wi_i*2]};
                    jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[31:0]);
                    jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[63:32]);
                    jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[95:64]);
                    jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[127:96]);
                end
                for (wi_i = 0; wi_i < 16; wi_i = wi_i + 1) begin
                    pw_word = gv_wgt[nt_i*16 + wi_i];
                    jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[31:0]);
                    jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[63:32]);
                    jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[95:64]);
                    jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[127:96]);
                end
            end

            // Configure L1 SDRAM stream: op_mode=0, w4=1, lut=1, T=16, K=32, N=16 (unused)
            jtag_send_cmd(OP_WRITE_REG, 24'd0, {3'd0, 8'd16, 8'd32, 8'd16, 2'd1, 1'b1, 2'd0});
            jtag_send_cmd(OP_WRITE_REG, 24'd1, {8'd64, 8'd0, 8'd0, 8'd16});
            jtag_send_cmd(OP_WRITE_REG, 24'd2, 32'd0);
            // reg_cfg3: dst_is_act=1 (bit29), act_stride=2, dst_stride=4
            jtag_send_cmd(OP_WRITE_REG, 24'd3, (32'd1 << 29) | (32'd2 << 21) | (32'd4 << 12));
            // reg_cfg4: stream_en=1, ntiles=4, kchunks=1, rd_lat=4, w_words=16
            jtag_send_cmd(OP_WRITE_REG, 24'd4, (32'd16 << 16) | (32'd4 << 13) | (32'd1 << 9) | (32'd4 << 1) | 32'd1);
            jtag_send_cmd(OP_WRITE_REG, 24'd5, 32'd256);
            jtag_send_cmd(OP_START, 24'd0, 32'd0);

            wait_done(tot_c, mac_c, epi_c, wr_c, csum_c);
            jtag_read_resp(16'h2009, stall_info);
            jtag_read_resp(16'h200B, dma_info);
            err_cnt = 0;
            for (i = 0; i < 64; i = i + 1) begin
                jtag_read_128b(4'h1, 8'd64 + i[7:0], got128);
                if (got128 !== gv_hpost[i]) begin
                    $display("FAIL Test 5a SDRAM DMA h_post[%0d]: got=%h exp=%h", i, got128, gv_hpost[i]);
                    err_cnt = err_cnt + 1;
                end
            end
            if (err_cnt != 0) $finish(1);
            $display("PASS Test 5a (FFN_W4A8 L1 SDRAM Ping-Pong 4 tiles): 0 LSB mismatch! tot=%0d mac=%0d dma_cyc=%0d stall_cyc=%0d",
                     tot_c, mac_c, dma_info[63:32], stall_info[63:32]);

            // Test 5b: Multi-K-Chunk (K_chunks=2, K=64 -> N=32, T=4, 2x2=4 chunks) with Ping-Pong + psum_mem
            for (t_i = 0; t_i < 4; t_i = t_i + 1) begin
                for (k16_i = 0; k16_i < 4; k16_i = k16_i + 1) begin
                    for (i = 0; i < 16; i = i + 1) begin
                        act_w[i*8 +: 8] = $signed(t_i + k16_i + i[1:0]); // values 0..9
                    end
                    jtag_write_128b(OP_COMMIT_ACT, (t_i*4 + k16_i) & 8'hFF, act_w);
                end
            end
            jtag_send_cmd(OP_SDRAM_CMD, 24'h000000, 32'd512); // SET_ADDR = 512
            for (nt_i = 0; nt_i < 2; nt_i = nt_i + 1) begin
                for (kc_i = 0; kc_i < 2; kc_i = kc_i + 1) begin
                    for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1) begin
                        // Pack 16 params: bias=5, m16=1, shift=0
                        jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, 32'd5);
                        jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, 32'h0000_0001);
                        jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, 32'd5);
                        jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, 32'h0000_0001);
                    end
                    for (row_i = 0; row_i < 16; row_i = row_i + 1) begin
                        w4_nib = ((nt_i + kc_i + row_i) % 3) + 1; // +1,+2,+3 in W4
                        for (wi_i = 0; wi_i < 16; wi_i = wi_i + 1) pw_word[wi_i*8 +: 8] = {w4_nib, w4_nib};
                        jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[31:0]);
                        jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[63:32]);
                        jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[95:64]);
                        jtag_send_cmd(OP_SDRAM_CMD, 24'h100000, pw_word[127:96]);
                    end
                end
            end

            // reg_cfg0: op_mode=0, w4=1, lut=0, T=4, K=32 (per chunk!), N=16
            jtag_send_cmd(OP_WRITE_REG, 24'd0, {3'd0, 8'd16, 8'd32, 8'd4, 2'd0, 1'b1, 2'd0});
            jtag_send_cmd(OP_WRITE_REG, 24'd1, 32'd0);
            jtag_send_cmd(OP_WRITE_REG, 24'd2, 32'd0);
            // reg_cfg3: dst_is_act=0, act_stride=4 (K=64/16), dst_stride=2 (N=32/16)
            jtag_send_cmd(OP_WRITE_REG, 24'd3, (32'd4 << 21) | (32'd2 << 12));
            // reg_cfg4: stream_en=1, ntiles=2, kchunks=2, rd_lat=4, w_words=16
            jtag_send_cmd(OP_WRITE_REG, 24'd4, (32'd16 << 16) | (32'd4 << 13) | (32'd2 << 9) | (32'd2 << 1) | 32'd1);
            jtag_send_cmd(OP_WRITE_REG, 24'd5, 32'd512);
            jtag_send_cmd(OP_START, 24'd0, 32'd0);

            wait_done(tot_c, mac_c, epi_c, wr_c, csum_c);
            jtag_read_resp(16'h2009, stall_info);
            jtag_read_resp(16'h200B, dma_info);
            err_cnt = 0;
            for (t_i = 0; t_i < 4; t_i = t_i + 1) begin
                for (nt_i = 0; nt_i < 2; nt_i = nt_i + 1) begin
                    jtag_read_128b(4'h0, (t_i*2 + nt_i) & 8'hFF, got128);
                    for (row_i = 0; row_i < 16; row_i = row_i + 1) begin
                        mac_acc = 5;
                        for (k16_i = 0; k16_i < 4; k16_i = k16_i + 1) begin
                            kc_i = k16_i / 2;
                            w4_nib = ((nt_i + kc_i + row_i) % 3) + 1;
                            for (i = 0; i < 16; i = i + 1) begin
                                mac_acc = mac_acc + (t_i + k16_i + (i % 4)) * w4_nib;
                            end
                        end
                        if (mac_acc > 127) mac_acc = 127;
                        if (mac_acc < -128) mac_acc = -128;
                        exp_w[row_i*8 +: 8] = mac_acc[7:0];
                    end
                    if (got128 !== exp_w) begin
                        $display("FAIL Test 5b K-chunk SDRAM DMA [t=%0d nt=%0d]: got=%h exp=%h", t_i, nt_i, got128, exp_w);
                        err_cnt = err_cnt + 1;
                    end
                end
            end
            if (err_cnt != 0) $finish(1);
            $display("PASS Test 5b (2x2 K-Chunk SDRAM Ping-Pong DMA): 0 LSB mismatch! tot=%0d mac=%0d dma_cyc=%0d stall_cyc=%0d",
                     tot_c, mac_c, dma_info[63:32], stall_info[63:32]);
        end

        $display("ALL RTL GOLDEN TESTS PASSED WITH 0 LSB MISMATCH!");
        $finish(0);
    end
endmodule
