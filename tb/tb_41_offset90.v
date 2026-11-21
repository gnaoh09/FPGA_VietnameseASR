`timescale 1ns/1ps
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
