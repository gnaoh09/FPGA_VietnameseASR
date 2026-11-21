#!/usr/bin/env python3
"""Simple macOS UART console helper for FPGA research designs."""

import argparse
import glob
import sys
import time


def find_default_port() -> str | None:
    patterns = ["/dev/cu.usbserial-*", "/dev/cu.usbmodem*", "/dev/cu.SLAB_USBtoUART*"]
    matches: list[str] = []
    for pat in patterns:
        matches.extend(sorted(glob.glob(pat)))
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Open serial console to FPGA UART")
    parser.add_argument("-p", "--port", help="Serial port path (auto-detected if omitted)")
    parser.add_argument("-b", "--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    args = parser.parse_args()

    try:
        import serial  # type: ignore
    except ImportError:
        print("pyserial is not installed. Install with: python3 -m pip install --user pyserial", file=sys.stderr)
        return 1

    port = args.port or find_default_port()
    if not port:
        print("No USB-UART device found under /dev/cu.usbserial-* or /dev/cu.usbmodem*.", file=sys.stderr)
        print("Connect a 3.3V USB-to-TTL adapter (CP2102/FT232/CH340) to the board header first.", file=sys.stderr)
        return 1

    print(f"Connecting to {port} @ {args.baud} baud (Ctrl-C to exit)...")
    with serial.Serial(port, args.baud, timeout=0.1) as ser:
        while True:
            data = ser.read(1024)
            if data:
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            time.sleep(0.01)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nClosed.")
        raise SystemExit(0)
