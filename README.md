# Cyclone IV (OMDAZZ EP4CE6E22C8) — macOS Research Workspace

Môi trường phát triển RTL & RISC-V trên macOS Apple Silicon cho board **OMDAZZ / RZ-EasyFPGA A2.2 (`EP4CE6E22C8`)**.

## Thực tế phần cứng trên board
- **FPGA:** Altera Cyclone IV `EP4CE6E22C8` (6,272 LEs, 270 Kbit M9K BRAM, 2 PLLs).
- **SDRAM (`U6`):** Chân hàn SDRAM trên bản board này đang bỏ trống (dùng BRAM nội cho thiết kế hiện tại).
- **Nguồn USB:** Cổng `USB PWR` trên board chỉ cấp nguồn 5V, không có chip USB-UART.
- **UART:** Board chỉ có cổng DB9 RS-232 (`U2`, `PIN_114`/`PIN_115`). Để giao tiếp hai chiều với Python trên macOS, cắm thêm **USB-to-TTL 3.3V** (CP2102/FT232/CH340) vào chân header `P1`/`P2`/`P3` + `GND`.

## Toolchain đang dùng
- **Simulation / Lint (native macOS):** `iverilog`, `verilator`
- **Synthesis / Place & Route (Docker `linux/amd64` + Rosetta 2):** `raetro/quartus:21.1.1`
- **JTAG Programmer (native macOS):** `openFPGALoader -c usb-blaster`
- **RISC-V Compiler (native macOS):** `riscv64-elf-gcc` (hỗ trợ `-march=rv32i -mabi=ilp32` / `rv32im`)

## Báo cáo Thực nghiệm ZipFormer-150M / 30M trên Cyclone IV
- Số liệu thực nghiệm & kịch bản tái lập được lưu trong `build/*.json` và `scripts/`.
- Chạy lại toàn bộ thực nghiệm G0 → G6: `bash scripts/run_all_experiments.sh`

## Các lệnh nhanh
```bash
make sim     # Chạy testbench RTL bằng Icarus Verilog
make lint    # Kiểm tra lint bằng Verilator
make build   # Biên dịch ra .sof và .svf bằng Quartus 21.1.1 trong Docker
make detect  # Kiểm tra chip qua USB-Blaster
make flash   # Nạp build/output_files/top.svf vào SRAM của FPGA
make uart    # Mở console UART (khi đã gắn USB-to-TTL)
```
