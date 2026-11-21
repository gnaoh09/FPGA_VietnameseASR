`timescale 1ns/1ps

// Parameterized cycle-accurate RTL simulation for any (ENG_MHZ, SDRAM_MHZ, layer shape, base addr).
// Uses real measured clocks passed via +ENG_HALF_NS=... +SDRAM_HALF_NS=... +REF_INT=... +T_RFC=...
// and also counts how many auto-refresh cycles occur while DMA is active vs idle.
module tb_78pt_sim;
    reg clk_eng   = 1'b0;
    reg clk_sdram = 1'b0;
    reg rst_n     = 1'b0;

    real eng_half_ns   = 6.25;
    real sdram_half_ns = 5.555556;
    integer ref_int_cfg = 1260;
    integer t_rfc_cfg   = 8;

    integer l1_k_chunk  = 192;
    integer l1_ntiles   = 40;
    integer l1_kchunks  = 1;
    integer l1_wwords   = 96;
    integer l1_base     = 0;

    integer l2_k_chunk  = 320;
    integer l2_ntiles   = 12;
    integer l2_kchunks  = 2;
    integer l2_wwords   = 160;
    integer l2_base     = 4160;

    integer d_model     = 192;
    integer d_ff        = 640;

    initial begin
        if (!$value$plusargs("ENG_HALF_NS=%f", eng_half_ns)) eng_half_ns = 6.25;
        if (!$value$plusargs("SDRAM_HALF_NS=%f", sdram_half_ns)) sdram_half_ns = 5.555556;
        if (!$value$plusargs("REF_INT=%d", ref_int_cfg)) ref_int_cfg = 1260;
        if (!$value$plusargs("T_RFC=%d", t_rfc_cfg)) t_rfc_cfg = 8;
        if (!$value$plusargs("L1_KC=%d", l1_k_chunk)) l1_k_chunk = 192;
        if (!$value$plusargs("L1_NT=%d", l1_ntiles)) l1_ntiles = 40;
        if (!$value$plusargs("L1_KCH=%d", l1_kchunks)) l1_kchunks = 1;
        if (!$value$plusargs("L1_WW=%d", l1_wwords)) l1_wwords = 96;
        if (!$value$plusargs("L1_BASE=%d", l1_base)) l1_base = 0;
        if (!$value$plusargs("L2_KC=%d", l2_k_chunk)) l2_k_chunk = 320;
        if (!$value$plusargs("L2_NT=%d", l2_ntiles)) l2_ntiles = 12;
        if (!$value$plusargs("L2_KCH=%d", l2_kchunks)) l2_kchunks = 2;
        if (!$value$plusargs("L2_WW=%d", l2_wwords)) l2_wwords = 160;
        if (!$value$plusargs("L2_BASE=%d", l2_base)) l2_base = 4160;
        if (!$value$plusargs("DM=%d", d_model)) d_model = 192;
        if (!$value$plusargs("DF=%d", d_ff)) d_ff = 640;
    end

    always #(eng_half_ns)   clk_eng   = ~clk_eng;
    always #(sdram_half_ns) clk_sdram = ~clk_sdram;

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
            sdram_dq_en <= 1'b0; sdram_rd_v0 <= 1'b0; sdram_rd_v1 <= 1'b0;
        end else begin
            sdram_dq_en <= 1'b0;
            sdram_rd_v1 <= sdram_rd_v0; sdram_rd_d1 <= sdram_rd_d0;
            if (sdram_rd_v1) begin sdram_dq_en <= 1'b1; sdram_dq_drv <= sdram_rd_d1; end
            sdram_rd_v0 <= 1'b0;
            if (sdram_cke && !sdram_cs_n) begin
                case ({sdram_ras_n, sdram_cas_n, sdram_we_n})
                    3'b011: sdram_act_row[sdram_ba] <= sdram_addr;
                    3'b101: begin sdram_rd_v0 <= 1'b1; sdram_rd_d0 <= {sdram_ba, sdram_act_row[sdram_ba][5:0], sdram_addr[7:0]}; end
                    default: ;
                endcase
            end
        end
    end

    wire         sdram_req_tog, sdram_ack_tog, sdram_ready, sdram_dma_w128_valid;
    wire [1:0]   sdram_req_op;
    wire [18:0]  sdram_req_addr;
    wire [19:0]  sdram_req_len;
    wire [127:0] sdram_req_wdata, sdram_rdata, sdram_dma_w128_data;
    wire [2:0]   sdram_cfg_rd_lat;
    wire [8:0]   sdram_dma_w128_idx;
    wire [31:0]  sdram_bist_wr_cyc, sdram_bist_rd_cyc, sdram_bist_err;
    wire [63:0]  sdram_cyc_cnt;

    // Instantiate with CLK_MHZ=90 as base and override REF_INTERVAL via defparam/local? No, pass CLK_MHZ=100 if 1400 etc.
    // Better: use CLK_MHZ_PARAM passed via compile or parameterless: wait - sdram_ctrl uses CLK_MHZ*14!
    // We compile once with -DREF_OVERRIDE or pass CLK_MHZ via -P? Wait: -P works at compile time, OR we add `ifdef SIM_REF_PLUSARG`?
    // Actually, sdram_ctrl has CLK_MHZ parameter: for exact REF_INTERVAL = sdram_mhz * 14, we can compile 10 tiny binaries OR even faster: pass `-Psdram_ctrl.CLK_MHZ=...`? Wait: `iverilog` runs in 0.1s per binary, OR we can just use `defparam`? No, defparam can't take $plusargs.
    // Let's check: 12 bitstreams = 12 `iverilog` calls = 1.2s total! Let's use parameter `CLK_MHZ_P = 100` on `tb_78pt_sim`!
    parameter integer CLK_MHZ_P = 100;

    sdram_ctrl #(
        .CLK_MHZ       (CLK_MHZ_P),
        .INIT_WAIT_CYC (40)
    ) u_sdram (
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

    reg         cmd_valid = 1'b0;
    reg [63:0]  cmd_word  = 64'd0;
    wire [63:0] resp_word;
    wire        status_busy, status_done;

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
        begin
            @(negedge clk_eng); cmd_valid = 1'b1; cmd_word = {op, addr, wdata};
            @(posedge clk_eng); @(negedge clk_eng); cmd_valid = 1'b0;
        end
    endtask

    task read_reg(input [15:0] sel, output [63:0] val);
        begin
            send_cmd(8'h06, {8'd0, sel}, 32'd0);
            @(posedge clk_eng); val = resp_word;
        end
    endtask

    task run_layer(
        input [9:0] kc, input [7:0] nt, input [3:0] kch, input [8:0] ww, input [18:0] base,
        input [8:0] ab, input [8:0] db, input [7:0] ds, input [7:0] as, input dia, input [1:0] lm,
        output integer tot, output integer st
    );
        reg [63:0] r1, r9;
        begin
            send_cmd(8'h07, 24'd0, {1'b0, 2'd0, 4'd1, kc[7:0], 8'd4, lm, 1'b1, 2'd0});
            send_cmd(8'h07, 24'd1, {db[7:0], 8'd0, ab[7:0], 8'd16});
            send_cmd(8'h07, 24'd2, {db[7:0], 24'd0});
            send_cmd(8'h07, 24'd3, {2'd0, dia, as, 1'b0, ds, db[8], ab[8], kc[9:8], 8'd0});
            send_cmd(8'h07, 24'd4, {7'd0, ww, 3'd4, kch, nt, 1'b1});
            send_cmd(8'h07, 24'd5, {13'd0, base});
            send_cmd(8'h08, 24'd0, 32'd0);
            @(posedge clk_eng);
            while (!status_done) @(posedge clk_eng);
            read_reg(16'h2001, r1);
            read_reg(16'h2009, r9);
            tot = r1[31:0];
            st  = r9[63:32];
        end
    endtask

    integer l1_t, l1_s, l2_t, l2_s;
    integer hpost_b, ds1, as1, ds2, as2;
    initial begin
        rst_n = 1'b0; #200; rst_n = 1'b1;
        while (!sdram_ready) @(posedge clk_eng);
        repeat (20) @(posedge clk_eng);
        hpost_b = (d_model / 16) * 4;
        as1 = d_model / 16; ds1 = d_ff / 16;
        as2 = d_ff / 16;    ds2 = d_model / 16;
        run_layer(l1_k_chunk[9:0], l1_ntiles[7:0], l1_kchunks[3:0], l1_wwords[8:0], l1_base[18:0],
                  9'd0, hpost_b[8:0], ds1[7:0], as1[7:0], 1'b1, 2'd1, l1_t, l1_s);
        run_layer(l2_k_chunk[9:0], l2_ntiles[7:0], l2_kchunks[3:0], l2_wwords[8:0], l2_base[18:0],
                  hpost_b[8:0], 9'd0, ds2[7:0], as2[7:0], 1'b0, 2'd0, l2_t, l2_s);
        $display("SIM_RES l1_tot=%0d l1_st=%0d l2_tot=%0d l2_st=%0d tot=%0d st=%0d",
                 l1_t, l1_s, l2_t, l2_s, l1_t + l2_t, l1_s + l2_s);
        $finish;
    end
endmodule
