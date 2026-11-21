`timescale 1ns/1ps

// Fast cycle-accurate RTL simulation of 192->640->192 (W4A8, T_tile=4)
// at 80 MHz Engine / 90 MHz SDRAM across offsets 0..31 and multiple refresh phases.
// Directly drives zipformer_engine cmd_valid/cmd_word (no slow JTAG bitbanging)
// and models HY57V641620 SDRAM command latency (CL=3, full 8 MiB address space).
module tb_offset_sim;
    parameter integer SDRAM_MHZ = 90;

    reg clk_80m   = 1'b0;
    reg clk_sdram = 1'b0;
    reg rst_n     = 1'b0;

    always #6.25 clk_80m = ~clk_80m; // 80.000 MHz
    generate
        if (SDRAM_MHZ == 120) begin : g120
            always #4.167 clk_sdram = ~clk_sdram;
        end else if (SDRAM_MHZ == 100) begin : g100
            always #5.000 clk_sdram = ~clk_sdram;
        end else if (SDRAM_MHZ == 90) begin : g90
            always #5.556 clk_sdram = ~clk_sdram;
        end else begin : g60
            always #8.333 clk_sdram = ~clk_sdram;
        end
    endgenerate

    // Behavioral HY57V641620 (CL=3, BL=1)
    wire        sdram_cke, sdram_cs_n, sdram_ras_n, sdram_cas_n, sdram_we_n;
    wire [1:0]  sdram_ba, sdram_dqm;
    wire [11:0] sdram_addr;
    wire [15:0] sdram_dq_out;
    wire        sdram_dq_oe;
    reg  [15:0] sdram_dq_drv;
    reg         sdram_dq_en;
    wire [15:0] sdram_dq = sdram_dq_oe ? sdram_dq_out : (sdram_dq_en ? sdram_dq_drv : 16'h0000);

    reg [11:0] sdram_act_row [0:3];
    reg        sdram_rd_v0, sdram_rd_v1;
    reg [15:0] sdram_rd_d0, sdram_rd_d1;

    always @(posedge clk_sdram) begin
        if (!rst_n) begin
            sdram_dq_en <= 1'b0;
            sdram_rd_v0 <= 1'b0;
            sdram_rd_v1 <= 1'b0;
        end else begin
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
                    3'b011: sdram_act_row[sdram_ba] <= sdram_addr;
                    3'b101: begin
                        sdram_rd_v0 <= 1'b1;
                        sdram_rd_d0 <= {sdram_ba, sdram_act_row[sdram_ba][5:0], sdram_addr[7:0]};
                    end
                    default: ;
                endcase
            end
        end
    end

    // Engine <-> SDRAM CDC wires
    wire         sdram_req_tog, sdram_ack_tog, sdram_ready, sdram_dma_w128_valid;
    wire [1:0]   sdram_req_op;
    wire [18:0]  sdram_req_addr;
    wire [19:0]  sdram_req_len;
    wire [127:0] sdram_req_wdata, sdram_rdata, sdram_dma_w128_data;
    wire [2:0]   sdram_cfg_rd_lat;
    wire [8:0]   sdram_dma_w128_idx;
    wire [31:0]  sdram_bist_wr_cyc, sdram_bist_rd_cyc, sdram_bist_err;
    wire [63:0]  sdram_cyc_cnt;

    sdram_ctrl #(
        .CLK_MHZ       (SDRAM_MHZ),
        .INIT_WAIT_CYC (40)
    ) u_sdram (
        .clk_sdram          (clk_sdram),
        .clk_eng            (clk_80m),
        .rst_n              (rst_n),
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

    reg         cmd_valid = 1'b0;
    reg [63:0]  cmd_word  = 64'd0;
    wire [63:0] resp_word;
    wire        status_busy, status_done;

    zipformer_engine #(
        .NUM_LANES  (16),
        .MAX_T_TILE (4)
    ) u_engine (
        .clk                  (clk_80m),
        .rst_n                (rst_n),
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

    localparam [7:0]
        OP_SET_READ_PTR = 8'h06,
        OP_WRITE_REG    = 8'h07,
        OP_START        = 8'h08;

    task send_cmd(input [7:0] op, input [23:0] addr, input [31:0] wdata);
        begin
            @(negedge clk_80m);
            cmd_valid = 1'b1;
            cmd_word  = {op, addr, wdata};
            @(posedge clk_80m);
            @(negedge clk_80m);
            cmd_valid = 1'b0;
        end
    endtask

    task read_reg(input [15:0] sel, output [63:0] val);
        begin
            send_cmd(OP_SET_READ_PTR, {8'd0, sel}, 32'd0);
            @(posedge clk_80m);
            val = resp_word;
        end
    endtask

    task run_layer(
        input [9:0]  k_chunk,
        input [7:0]  n_tiles,
        input [3:0]  k_chunks,
        input [8:0]  w_words,
        input [18:0] sdram_base,
        input [8:0]  act_base,
        input [8:0]  dst_base,
        input [7:0]  dst_stride,
        input [7:0]  act_stride,
        input        dst_is_act,
        input [1:0]  lut_mode,
        output integer tot_cyc,
        output integer stall_cyc,
        output integer dma_cyc,
        output integer mac_cyc
    );
        reg [63:0] r1, r9, r11;
        begin
            send_cmd(OP_WRITE_REG, 24'd0, {1'b0, 2'd0, 4'd1, k_chunk[7:0], 8'd4, lut_mode, 1'b1, 2'd0});
            send_cmd(OP_WRITE_REG, 24'd1, {dst_base[7:0], 8'd0, act_base[7:0], 8'd16});
            send_cmd(OP_WRITE_REG, 24'd2, {dst_base[7:0], 24'd0});
            send_cmd(OP_WRITE_REG, 24'd3, {2'd0, dst_is_act, act_stride, 1'b0, dst_stride, dst_base[8], act_base[8], k_chunk[9:8], 8'd0});
            send_cmd(OP_WRITE_REG, 24'd4, {7'd0, w_words, 3'd4, k_chunks, n_tiles, 1'b1});
            send_cmd(OP_WRITE_REG, 24'd5, {13'd0, sdram_base});
            send_cmd(OP_START, 24'd0, 32'd0);
            @(posedge clk_80m);
            while (!status_done) @(posedge clk_80m);
            read_reg(16'h2001, r1);
            read_reg(16'h2009, r9);
            read_reg(16'h200B, r11);
            tot_cyc   = r1[31:0];
            mac_cyc   = r1[63:32];
            stall_cyc = r9[63:32];
            dma_cyc   = r11[63:32];
        end
    endtask

    integer off, ph;
    integer l1_tot, l1_st, l1_dma, l1_mac;
    integer l2_tot, l2_st, l2_dma, l2_mac;

    initial begin
        rst_n = 1'b0;
        #200;
        rst_n = 1'b1;
        while (!sdram_ready) @(posedge clk_80m);
        repeat (20) @(posedge clk_80m);

        // 1) Sweep offsets 0..31 on 192->640->192 (L1 base=16384+off, L2 base=32768+off)
        $display("=== SIM SWEEP OFFSET 0..31 (192->640->192 @ 80/%0d MHz) ===", SDRAM_MHZ);
        for (off = 0; off < 32; off = off + 1) begin
            run_layer(10'd192, 8'd40, 4'd1, 9'd96, 19'd16384 + off, 9'd0, 9'd48, 8'd40, 8'd12, 1'b1, 2'd1, l1_tot, l1_st, l1_dma, l1_mac);
            run_layer(10'd320, 8'd12, 4'd2, 9'd160, 19'd32768 + off, 9'd48, 9'd0, 8'd12, 8'd40, 1'b0, 2'd0, l2_tot, l2_st, l2_dma, l2_mac);
            $display("OFF=%02d | L1 tot=%0d st=%0d dma=%0d | L2 tot=%0d st=%0d dma=%0d | SUM=%0d",
                     off, l1_tot, l1_st, l1_dma, l2_tot, l2_st, l2_dma, l1_tot + l2_tot);
        end

        // 2) Refresh-phase sweep at offset 1 vs offset 15 (insert controlled delay 0..1200 eng cycles)
        $display("=== SIM PHASE SWEEP (offset 1 vs 15, delay 0..1000 step 100 eng cyc) ===");
        for (ph = 0; ph <= 1000; ph = ph + 100) begin
            repeat (ph) @(posedge clk_80m);
            run_layer(10'd192, 8'd40, 4'd1, 9'd96, 19'd16385, 9'd0, 9'd48, 8'd40, 8'd12, 1'b1, 2'd1, l1_tot, l1_st, l1_dma, l1_mac);
            run_layer(10'd320, 8'd12, 4'd2, 9'd160, 19'd32769, 9'd48, 9'd0, 8'd12, 8'd40, 1'b0, 2'd0, l2_tot, l2_st, l2_dma, l2_mac);
            $display("PH=%04d OFF=01 | L1 tot=%0d st=%0d dma=%0d | L2 tot=%0d st=%0d dma=%0d | SUM=%0d",
                     ph, l1_tot, l1_st, l1_dma, l2_tot, l2_st, l2_dma, l1_tot + l2_tot);
            repeat (ph) @(posedge clk_80m);
            run_layer(10'd192, 8'd40, 4'd1, 9'd96, 19'd16399, 9'd0, 9'd48, 8'd40, 8'd12, 1'b1, 2'd1, l1_tot, l1_st, l1_dma, l1_mac);
            run_layer(10'd320, 8'd12, 4'd2, 9'd160, 19'd32783, 9'd48, 9'd0, 8'd12, 8'd40, 1'b0, 2'd0, l2_tot, l2_st, l2_dma, l2_mac);
            $display("PH=%04d OFF=15 | L1 tot=%0d st=%0d dma=%0d | L2 tot=%0d st=%0d dma=%0d | SUM=%0d",
                     ph, l1_tot, l1_st, l1_dma, l2_tot, l2_st, l2_dma, l1_tot + l2_tot);
        end
        $finish;
    end
endmodule
