`timescale 1ns/1ps

// JTAG Bridge using Altera sld_virtual_jtag (in synthesis) and a behavioral TAP (in simulation).
//   vir_in == 0: 64-bit command/write frame shifted in Virtual Shift-DR, latched on Virtual Update-DR.
//   vir_in == 1: 64-bit resp_word captured in Virtual Capture-DR and shifted out on TDO in Virtual Shift-DR.
module jtag_bridge (
    input  wire        clk,
    input  wire        rst_n,
    // Simulation-only JTAG pins (unused in hardware synthesis)
    input  wire        sim_tck,
    input  wire        sim_tms,
    input  wire        sim_tdi,
    output wire        sim_tdo,
    // Command output in clk (50 MHz) domain
    output reg         cmd_valid,
    output reg  [63:0] cmd_word,
    // Response word held steady in clk domain before read scan
    input  wire [63:0] resp_word
);
    wire tckutap;
    wire tdiutap;
    wire v_cdr;
    wire v_sdr;
    wire v_udr;
    wire vir_in;
    wire tdouser;

`ifdef SIMULATION
    // Behavioral TAP model supporting USER0 (10'h00C) and USER1 (10'h00E) for Icarus/Verilator sim.
    localparam [9:0] IR_USER0 = 10'h00C;
    localparam [9:0] IR_USER1 = 10'h00E;

    localparam [3:0]
        TLR  = 4'd0,  RTI  = 4'd1,
        SDRS = 4'd2,  CDR  = 4'd3,  SDR = 4'd4,  E1DR = 4'd5, PDR = 4'd6, E2DR = 4'd7, UDR = 4'd8,
        SIRS = 4'd9,  CIR  = 4'd10, SIR = 4'd11, E1IR = 4'd12, PIR = 4'd13, E2IR = 4'd14, UIR = 4'd15;

    reg [3:0] tap_state;
    reg [9:0] ir_sr;
    reg [9:0] ir_hold;
    reg       tdo_ir;

    always @(posedge sim_tck or negedge rst_n) begin
        if (!rst_n) begin
            tap_state <= TLR;
            ir_sr     <= 10'd0;
            ir_hold   <= 10'h006; // IDCODE default
            tdo_ir    <= 1'b0;
        end else begin
            case (tap_state)
                TLR:  tap_state <= sim_tms ? TLR  : RTI;
                RTI:  tap_state <= sim_tms ? SDRS : RTI;
                SDRS: tap_state <= sim_tms ? SIRS : CDR;
                CDR:  tap_state <= sim_tms ? E1DR : SDR;
                SDR:  tap_state <= sim_tms ? E1DR : SDR;
                E1DR: tap_state <= sim_tms ? UDR  : PDR;
                PDR:  tap_state <= sim_tms ? E2DR : PDR;
                E2DR: tap_state <= sim_tms ? UDR  : SDR;
                UDR:  tap_state <= sim_tms ? SDRS : RTI;
                SIRS: tap_state <= sim_tms ? TLR  : CIR;
                CIR:  tap_state <= sim_tms ? E1IR : SIR;
                SIR:  tap_state <= sim_tms ? E1IR : SIR;
                E1IR: tap_state <= sim_tms ? UIR  : PIR;
                PIR:  tap_state <= sim_tms ? E2IR : PIR;
                E2IR: tap_state <= sim_tms ? UIR  : SIR;
                UIR:  tap_state <= sim_tms ? SDRS : RTI;
                default: tap_state <= TLR;
            endcase

            if (tap_state == CIR) begin
                ir_sr <= 10'b0000000001;
            end else if (tap_state == SIR) begin
                tdo_ir <= ir_sr[0];
                ir_sr  <= {sim_tdi, ir_sr[9:1]};
            end else if (tap_state == UIR) begin
                ir_hold <= ir_sr;
            end
        end
    end

    wire user_active = (ir_hold == IR_USER0) || (ir_hold == IR_USER1);
    assign tckutap = sim_tck;
    assign tdiutap = sim_tdi;
    assign v_cdr   = user_active && (tap_state == CDR);
    assign v_sdr   = user_active && (tap_state == SDR);
    assign v_udr   = user_active && (tap_state == UDR);
    assign vir_in  = (ir_hold == IR_USER1);
    assign sim_tdo = (tap_state == SIR) ? tdo_ir : tdouser;
`else
    wire [0:0] vir_bus;
    assign vir_in = vir_bus[0];

    sld_virtual_jtag #(
        .sld_auto_instance_index ("NO"),
        .sld_instance_index      (0),
        .sld_ir_width            (1)
    ) u_vtag (
        .tck               (tckutap),
        .tdi               (tdiutap),
        .ir_in             (vir_bus),
        .tdo               (tdouser),
        .ir_out            (1'b0),
        .virtual_state_cdr (v_cdr),
        .virtual_state_sdr (v_sdr),
        .virtual_state_udr (v_udr)
    );
    assign sim_tdo = 1'b0;
`endif

    // JTAG TCK domain shift registers
    reg [63:0] cmd_sr;
    reg [63:0] resp_sr;

    always @(posedge tckutap or negedge rst_n) begin
        if (!rst_n) begin
            cmd_sr  <= 64'd0;
            resp_sr <= 64'd0;
        end else if (!vir_in) begin
            if (v_sdr) begin
                cmd_sr <= {tdiutap, cmd_sr[63:1]};
            end
        end else begin
            if (v_cdr) begin
                resp_sr <= resp_word;
            end else if (v_sdr) begin
                resp_sr <= {1'b0, resp_sr[63:1]};
            end
        end
    end

    assign tdouser = resp_sr[0];

    // Synchronize virtual Update-DR into clk (50 MHz) domain
    reg upd_s0, upd_s1, upd_s2;
    reg vir_s0, vir_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            upd_s0    <= 1'b0;
            upd_s1    <= 1'b0;
            upd_s2    <= 1'b0;
            vir_s0    <= 1'b0;
            vir_s1    <= 1'b0;
            cmd_valid <= 1'b0;
            cmd_word  <= 64'd0;
        end else begin
            upd_s0    <= v_udr;
            upd_s1    <= upd_s0;
            upd_s2    <= upd_s1;
            vir_s0    <= vir_in;
            vir_s1    <= vir_s0;
            cmd_valid <= 1'b0;
            if (upd_s1 && !upd_s2 && !vir_s1) begin
                cmd_valid <= 1'b1;
                cmd_word  <= cmd_sr;
            end
        end
    end
endmodule
