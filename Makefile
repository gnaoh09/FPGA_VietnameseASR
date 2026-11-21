PROJ        := top
TOP_MODULE  := top
BUILD_DIR   := build
RTL_SRCS    := $(wildcard rtl/*.v rtl/*.sv)
TB_TOP      := tb/tb_top.v
SVF_FILE    := $(BUILD_DIR)/output_files/$(PROJ).svf

.PHONY: all sim lint build flash detect uart clean

all: sim

sim: $(RTL_SRCS) $(TB_TOP)
	@mkdir -p $(BUILD_DIR)
	iverilog -g2012 -Wall -DSIMULATION -o $(BUILD_DIR)/tb_top.vvp $(RTL_SRCS) $(TB_TOP)
	vvp $(BUILD_DIR)/tb_top.vvp

lint: $(RTL_SRCS)
	verilator --lint-only -Wall -DSIMULATION --top-module $(TOP_MODULE) $(RTL_SRCS)

build:
	PROJ_NAME=$(PROJ) TOP_MODULE=$(TOP_MODULE) bash scripts/build_quartus.sh

flash: $(SVF_FILE)
	openFPGALoader -c usb-blaster $(SVF_FILE)

detect:
	openFPGALoader -c usb-blaster --detect

uart:
	python3 scripts/uart_console.py

clean:
	rm -rf $(BUILD_DIR)
