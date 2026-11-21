#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "=== [G0] Board & Toolchain Check ==="
openFPGALoader -c usb-blaster --detect

echo "=== [G1] ARM64 Baseline Benchmark ==="
python3 scripts/g1_arm_baseline.py

echo "=== [G2] Quantization Pareto & Noise Sweep ==="
python3 scripts/g2_quantization_pareto.py

echo "=== [G3] Golden Vectors, Lint, RTL Simulation & Quartus Synthesis ==="
python3 scripts/gen_golden_vectors.py
make lint
make sim
SDRAM_MHZ=100 make build

echo "=== [G4] Flash Bitstream & Run Real-Board Block Verification ==="
make flash
python3 scripts/fpga_jtag_runner.py

echo "=== [G4b] Real Physical SDRAM HY57V641620FTP-H (8 MiB) BIST + 512->1536->512 Ping-Pong DMA ==="
.venv/bin/python scripts/run_g4b_sdram_experiment.py --mhz 100

echo "=== [G4-Full] Real-Size 512->1536->512 Streaming FFN via JTAG + Memory Analysis ==="
.venv/bin/python scripts/run_ffn_512_1536_real.py
.venv/bin/python scripts/analyze_memory_bandwidth.py

echo "=== [G5b] Full 36-FFN Zipformer-30M Resident in 8 MiB SDRAM (Real VIVOS Activations, 0 LSB) ==="
.venv/bin/python scripts/run_g5b_30m_resident_sdram.py
.venv/bin/python scripts/run_g5b_multisent_fullseq.py

echo "=== [G5-G6] Schedule Compiler, 10x Hardware Runs & Power/Energy Evaluation ==="
python3 scripts/g5_g6_system_eval.py

echo "=== [G2-30M] Build FPGA Bit-Exact 30M ONNX + WER/FLEURS/Noise + Paired Bootstrap ==="
.venv/bin/python scripts/build_fpga_bitexact_30m_onnx.py
.venv/bin/python scripts/eval_30m_wer_pareto_noise.py

echo "=== ALL GATES G0..G6 + G4b/G5b SDRAM + REAL 512->1536->512 + 30M WER COMPLETED ==="
