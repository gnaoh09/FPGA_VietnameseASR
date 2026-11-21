# 50 MHz base clock on PIN_23
create_clock -name clk -period 20.000 [get_ports {clk}]

derive_pll_clocks
derive_clock_uncertainty

# Explicit asynchronous isolation across Engine (80MHz c0), SDRAM (c1/c2), base clock, and JTAG TCK
set_clock_groups -asynchronous \
    -group {clk u_pll|altpll_component|auto_generated|pll1|clk[0]} \
    -group {u_pll|altpll_component|auto_generated|pll1|clk[1] u_pll|altpll_component|auto_generated|pll1|clk[2]} \
    -group {altera_reserved_tck}

# Asynchronous board buttons, status LEDs, JTAG pins, and external SDRAM I/O (board-level phase shifted via PLL c2)
set_false_path -from [get_ports {rst_n key[*] altera_reserved_tdi altera_reserved_tms sdram_dq[*]}]
set_false_path -to [get_ports {led[*] altera_reserved_tdo sdram_clk sdram_cke sdram_cs_n sdram_ras_n sdram_cas_n sdram_we_n sdram_ba[*] sdram_addr[*] sdram_dqm[*] sdram_dq[*]}]
