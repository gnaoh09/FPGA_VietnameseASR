# 50 MHz base clock on PIN_23
create_clock -name clk -period 20.000 [get_ports {clk}]

derive_pll_clocks
derive_clock_uncertainty

# Asynchronous board buttons and status LEDs
set_false_path -from [get_ports {rst_n key[*]}]
set_false_path -to [get_ports {led[*]}]
