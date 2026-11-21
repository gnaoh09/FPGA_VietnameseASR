`timescale 1ns/1ps

// Multi-clock Cyclone IV E Hardware PLL (from 50 MHz PIN_23):
//   c0: 80.00 MHz (0 ps) -> jtag_bridge + zipformer_engine
//   c1: SDRAM_MHZ (80.00 MHz or 100.00 MHz, 0 ps) -> sdram_ctrl
//   c2: SDRAM_MHZ with negative board phase shift (-3000 ps @ 100 MHz / -3125 ps @ 80 MHz) -> PIN_43 sdram_clk
module pll_80m #(
    parameter integer ENG_MHZ   = 80,
    parameter integer SDRAM_MHZ = 100
) (
    input  wire inclk0,
    input  wire areset,
    output wire c0,
    output wire c1,
    output wire c2,
    output wire locked
);
`ifdef SIMULATION
    /* verilator lint_off PROCASSINIT */
    reg sim_clk80 = 1'b0;
    generate
        if (ENG_MHZ == 90) begin : gen_eng_90
            always #5.556 sim_clk80 = ~sim_clk80;
        end else if (ENG_MHZ == 70) begin : gen_eng_70
            always #7.143 sim_clk80 = ~sim_clk80;
        end else if (ENG_MHZ == 60) begin : gen_eng_60
            always #8.333 sim_clk80 = ~sim_clk80;
        end else begin : gen_eng_80
            always #6.25 sim_clk80 = ~sim_clk80;
        end
    endgenerate

    reg sim_clk_sdram = 1'b0;
    generate
        if (SDRAM_MHZ == 120) begin : gen_sim_120
            always #4.167 sim_clk_sdram = ~sim_clk_sdram;
        end else if (SDRAM_MHZ == 100) begin : gen_sim_100
            always #5.00 sim_clk_sdram = ~sim_clk_sdram;
        end else if (SDRAM_MHZ == 90) begin : gen_sim_90
            always #5.556 sim_clk_sdram = ~sim_clk_sdram;
        end else if (SDRAM_MHZ == 60) begin : gen_sim_60
            always #8.333 sim_clk_sdram = ~sim_clk_sdram;
        end else if (SDRAM_MHZ == 65) begin : gen_sim_65
            always #7.692 sim_clk_sdram = ~sim_clk_sdram;
        end else begin : gen_sim_80
            always #6.25 sim_clk_sdram = ~sim_clk_sdram;
        end
    endgenerate
    /* verilator lint_on PROCASSINIT */
    assign c0     = sim_clk80;
    assign c1     = sim_clk_sdram;
    assign c2     = sim_clk_sdram;
    assign locked = ~areset;
    wire unused_inclk = inclk0;
`else
    localparam integer ENG_MUL =
        (ENG_MHZ == 90) ? 9 :
        (ENG_MHZ == 70) ? 7 :
        (ENG_MHZ == 60) ? 6 : 8;
    localparam integer ENG_DIV = 5;

    localparam integer SDRAM_MUL =
        (SDRAM_MHZ == 120) ? 12 :
        (SDRAM_MHZ == 110) ? 11 :
        (SDRAM_MHZ == 100) ? 2  :
        (SDRAM_MHZ == 96)  ? 48 :
        (SDRAM_MHZ == 90)  ? 9  :
        (SDRAM_MHZ == 88)  ? 44 :
        (SDRAM_MHZ == 75)  ? 3  :
        (SDRAM_MHZ == 70)  ? 7  :
        (SDRAM_MHZ == 65)  ? 13 :
        (SDRAM_MHZ == 60)  ? 6  :
        (SDRAM_MHZ == 50)  ? 1  :
        (SDRAM_MHZ == 40)  ? 4  : 8;
    localparam integer SDRAM_DIV =
        (SDRAM_MHZ == 100) ? 1  :
        (SDRAM_MHZ == 96)  ? 25 :
        (SDRAM_MHZ == 88)  ? 25 :
        (SDRAM_MHZ == 75)  ? 2  :
        (SDRAM_MHZ == 65)  ? 10 :
        (SDRAM_MHZ == 50)  ? 1  : 5;
    localparam SDRAM_PHASE =
        (SDRAM_MHZ >= 110) ? "-2500" :
        (SDRAM_MHZ >= 88)  ? "-3000" :
        (SDRAM_MHZ >= 65)  ? "-3125" : "-3500";

    wire [4:0] sub_wire_clk;
    wire       sub_wire_locked;
    assign c0     = sub_wire_clk[0];
    assign c1     = sub_wire_clk[1];
    assign c2     = sub_wire_clk[2];
    assign locked = sub_wire_locked;

    altpll #(
        .bandwidth_type          ("AUTO"),
        .clk0_divide_by          (ENG_DIV),
        .clk0_duty_cycle         (50),
        .clk0_multiply_by        (ENG_MUL),
        .clk0_phase_shift        ("0"),
        .clk1_divide_by          (SDRAM_DIV),
        .clk1_duty_cycle         (50),
        .clk1_multiply_by        (SDRAM_MUL),
        .clk1_phase_shift        ("0"),
        .clk2_divide_by          (SDRAM_DIV),
        .clk2_duty_cycle         (50),
        .clk2_multiply_by        (SDRAM_MUL),
        .clk2_phase_shift        (SDRAM_PHASE),
        .compensate_clock        ("CLK0"),
        .inclk0_input_frequency  (20000),
        .intended_device_family  ("Cyclone IV E"),
        .lpm_hint                ("CBX_MODULE_PREFIX=pll_80m"),
        .lpm_type                ("altpll"),
        .operation_mode          ("NORMAL"),
        .pll_type                ("AUTO"),
        .port_activeclock        ("PORT_UNUSED"),
        .port_areset             ("PORT_USED"),
        .port_clkbad0            ("PORT_UNUSED"),
        .port_clkbad1            ("PORT_UNUSED"),
        .port_clkloss            ("PORT_UNUSED"),
        .port_clkswitch          ("PORT_UNUSED"),
        .port_configupdate       ("PORT_UNUSED"),
        .port_fbin               ("PORT_UNUSED"),
        .port_inclk0             ("PORT_USED"),
        .port_inclk1             ("PORT_UNUSED"),
        .port_locked             ("PORT_USED"),
        .port_pfdena             ("PORT_UNUSED"),
        .port_phasecounterselect ("PORT_UNUSED"),
        .port_phasedone          ("PORT_UNUSED"),
        .port_phasestep          ("PORT_UNUSED"),
        .port_phaseupdown        ("PORT_UNUSED"),
        .port_pllena             ("PORT_UNUSED"),
        .port_scanaclr           ("PORT_UNUSED"),
        .port_scanclk            ("PORT_UNUSED"),
        .port_scanclkena         ("PORT_UNUSED"),
        .port_scandata           ("PORT_UNUSED"),
        .port_scandataout        ("PORT_UNUSED"),
        .port_scandone           ("PORT_UNUSED"),
        .port_scanread           ("PORT_UNUSED"),
        .port_scanwrite          ("PORT_UNUSED"),
        .port_clk0               ("PORT_USED"),
        .port_clk1               ("PORT_USED"),
        .port_clk2               ("PORT_USED"),
        .port_clk3               ("PORT_UNUSED"),
        .port_clk4               ("PORT_UNUSED"),
        .port_clk5               ("PORT_UNUSED"),
        .port_clkena0            ("PORT_UNUSED"),
        .port_clkena1            ("PORT_UNUSED"),
        .port_clkena2            ("PORT_UNUSED"),
        .port_clkena3            ("PORT_UNUSED"),
        .port_clkena4            ("PORT_UNUSED"),
        .port_clkena5            ("PORT_UNUSED"),
        .port_extclk0            ("PORT_UNUSED"),
        .port_extclk1            ("PORT_UNUSED"),
        .port_extclk2            ("PORT_UNUSED"),
        .port_extclk3            ("PORT_UNUSED"),
        .self_reset_on_loss_lock ("OFF"),
        .width_clock             (5)
    ) altpll_component (
        .areset (areset),
        .inclk  ({1'b0, inclk0}),
        .clk    (sub_wire_clk),
        .locked (sub_wire_locked)
    );
`endif
endmodule
