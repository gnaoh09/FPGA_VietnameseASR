#!/usr/bin/env python3
"""G4 Real-Hardware Runner over Altera USB-Blaster (libftdi1) & sld_virtual_jtag.
Uploads golden vectors directly into Cyclone IV EP4CE10E22C8N M9K BRAM,
executes FFN W8A8 (Mode V & Mode S), FFN W4A8 (W4 unpack), and Joiner W8A8,
reads back outputs & hardware performance counters, and verifies 0 LSB mismatch.
"""

import ctypes
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT_DIR / "golden_vectors"
BUILD_DIR = ROOT_DIR / "build"

# JTAG & SLD Virtual JTAG Constants (from build/output_files/top.jdi: hub_ir_width=4, ir_width=1, node_addr=1)
IR_USER0 = 0x00C
IR_USER1 = 0x00E
VIR_LEN = 5
NODE_MASK = 1 << (VIR_LEN - 1)  # 0b10000 selects instance_id=0

# Engine Opcodes
OP_STAGE_W32 = 0x01
OP_COMMIT_ACT = 0x02
OP_COMMIT_WGT = 0x03
OP_WRITE_PARAM = 0x04
OP_WRITE_LUT = 0x05
OP_SET_READ_PTR = 0x06
OP_WRITE_REG = 0x07
OP_START = 0x08
OP_SDRAM_CMD = 0x09

# OP_SDRAM_CMD sub-ops (addr[23:20])
SDRAM_SUB_SET_ADDR = 0x0 << 20
SDRAM_SUB_PUSH_AUTO = 0x1 << 20
SDRAM_SUB_WR128 = 0x2 << 20
SDRAM_SUB_RD128 = 0x3 << 20
SDRAM_SUB_BIST = 0x4 << 20

# USB-Blaster FTDI bitbang base: bit3=nCS(1), bit5=LED(1)
BLASTER_BASE = 0x28


class UsbBlasterJtag:
    """Direct macOS USB-Blaster driver via libftdi1.dylib + sld_virtual_jtag."""

    def __init__(self, vid: int = 0x09FB, pid: int = 0x6001):
        self.ftdi = ctypes.CDLL("/opt/homebrew/lib/libftdi1.dylib")
        self.ftdi.ftdi_new.restype = ctypes.c_void_p
        self.ctx = ctypes.c_void_p(self.ftdi.ftdi_new())
        if not self.ctx:
            raise RuntimeError("ftdi_new() failed")
        rc = self.ftdi.ftdi_usb_open(self.ctx, vid, pid)
        if rc != 0:
            raise RuntimeError(f"ftdi_usb_open(0x{vid:04x}:0x{pid:04x}) failed with rc={rc}")
        self.ftdi.ftdi_set_latency_timer(self.ctx, 1)
        self.ftdi.ftdi_tcioflush(self.ctx)
        self._buf = bytearray()
        self._current_ir = None
        self._current_vir = None
        self.reset_to_rti()

    def close(self) -> None:
        if self.ctx:
            self._write_raw(bytes([BLASTER_BASE]))
            self.ftdi.ftdi_usb_close(self.ctx)
            self.ftdi.ftdi_free(self.ctx)
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _write_raw(self, data: bytes) -> None:
        chunk_sz = 4096
        mv = memoryview(data)
        for off in range(0, len(mv), chunk_sz):
            sub = bytes(mv[off : off + chunk_sz])
            rc = self.ftdi.ftdi_write_data(self.ctx, sub, len(sub))
            if rc != len(sub):
                raise RuntimeError(f"ftdi_write_data wrote {rc}/{len(sub)} bytes")

    def _read_raw(self, n_reads: int) -> list[int]:
        out = (ctypes.c_ubyte * n_reads)()
        got = 0
        t_start = time.monotonic()
        while got < n_reads:
            r = self.ftdi.ftdi_read_data(self.ctx, ctypes.byref(out, got), n_reads - got)
            if r > 0:
                got += r
            elif r < 0:
                raise RuntimeError(f"ftdi_read_data error {r}")
            elif time.monotonic() - t_start > 5.0:
                raise TimeoutError(f"Timed out waiting for {n_reads - got} JTAG read bytes")
        return [out[i] & 1 for i in range(n_reads)]

    @staticmethod
    def _pulse(tms: int, tdi: int = 0, read: bool = False) -> bytes:
        b0 = BLASTER_BASE | (0x02 if tms else 0) | (0x10 if tdi else 0)
        b1 = b0 | 0x01
        return bytes([b0 | 0x40, b1]) if read else bytes([b0, b1])

    def _queue_ir(self, ir10: int) -> None:
        if self._current_ir == ir10:
            return
        b = self._buf
        b.extend(self._pulse(1))
        b.extend(self._pulse(1))
        b.extend(self._pulse(0))
        b.extend(self._pulse(0))
        for i in range(10):
            b.extend(self._pulse(1 if i == 9 else 0, (ir10 >> i) & 1))
        b.extend(self._pulse(1))
        b.extend(self._pulse(0))
        self._current_ir = ir10

    def _queue_dr(self, nbits: int, val: int, read: bool = False) -> None:
        b = self._buf
        b.extend(self._pulse(1))
        b.extend(self._pulse(0))
        b.extend(self._pulse(0))
        if not read and nbits >= 16 and (nbits - 1) >= 8:
            # Drop TCK=0 before entering USB-Blaster Byte-Shift mode (prevents missing bit 0)
            b.append(BLASTER_BASE)
            n_bytes = (nbits - 1) // 8
            n_tail = nbits - n_bytes * 8
            # Process in chunks <= 63 bytes (USB-Blaster byte-shift header 0x80 | count)
            bit_pos = 0
            rem = n_bytes
            while rem > 0:
                cnt = min(63, rem)
                b.append(0x80 | cnt)
                for _ in range(cnt):
                    b.append((val >> bit_pos) & 0xFF)
                    bit_pos += 8
                rem -= cnt
            for i in range(n_tail):
                b.extend(self._pulse(1 if i == n_tail - 1 else 0, (val >> (bit_pos + i)) & 1, False))
        else:
            for i in range(nbits):
                b.extend(self._pulse(1 if i == nbits - 1 else 0, (val >> i) & 1, read))
        b.extend(self._pulse(1))
        b.extend(self._pulse(0))

    def flush(self) -> None:
        if self._buf:
            self._buf.append(BLASTER_BASE)
            self._write_raw(bytes(self._buf))
            self._buf.clear()

    def reset_to_rti(self) -> None:
        for _ in range(5):
            self._buf.extend(self._pulse(1))
        self._buf.extend(self._pulse(0))
        self._current_ir = None
        self._current_vir = None
        self.flush()

    def _ensure_vir(self, vir_val: int) -> None:
        if self._current_vir != vir_val:
            self._queue_ir(IR_USER1)
            self._queue_dr(VIR_LEN, NODE_MASK | vir_val, read=False)
            self._current_vir = vir_val
        self._queue_ir(IR_USER0)

    def queue_cmd(self, op: int, addr: int, wdata: int) -> None:
        self._ensure_vir(0)
        frame = ((op & 0xFF) << 56) | ((addr & 0xFFFFFF) << 32) | (wdata & 0xFFFFFFFF)
        self._queue_dr(64, frame, read=False)

    def send_cmd(self, op: int, addr: int, wdata: int) -> None:
        self.queue_cmd(op, addr, wdata)
        self.flush()

    def read_resp(self, sel: int) -> int:
        self.queue_cmd(OP_SET_READ_PTR, sel & 0xFFFF, 0)
        self._ensure_vir(1)
        self._queue_dr(64, 0, read=True)
        self.flush()
        bits = self._read_raw(64)
        return sum((bits[i] << i) for i in range(64))

    def queue_write_128b(self, commit_op: int, word_addr: int, word_val: int) -> None:
        for idx in range(4):
            self.queue_cmd(OP_STAGE_W32, idx, (word_val >> (32 * idx)) & 0xFFFFFFFF)
        self.queue_cmd(commit_op, word_addr & 0x1FF, 0)

    def queue_write_64b_param(self, param_addr: int, word_val: int) -> None:
        self.queue_cmd(OP_STAGE_W32, 0, word_val & 0xFFFFFFFF)
        self.queue_cmd(OP_STAGE_W32, 1, (word_val >> 32) & 0xFFFFFFFF)
        self.queue_cmd(OP_WRITE_PARAM, param_addr & 0x7F, 0)

    def read_128b(self, mem_id: int, word_addr: int) -> int:
        addr_hi = ((word_addr >> 8) & 3) << 9
        lo = self.read_resp(((mem_id & 0xF) << 12) | addr_hi | (0 << 8) | (word_addr & 0xFF))
        hi = self.read_resp(((mem_id & 0xF) << 12) | addr_hi | (1 << 8) | (word_addr & 0xFF))
        return lo | (hi << 64)

    def wait_sdram_idle(self, timeout_s: float = 5.0) -> int:
        t0 = time.monotonic()
        while True:
            st = self.read_resp(0x2005)
            if not ((st >> 36) & 1):
                return st
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"SDRAM controller busy timeout (0x2005=0x{st:016x})")

    def sdram_write_bulk_words(self, start_w128: int, words_128b: list[int], progress_cb=None) -> float:
        """Fast 4-cmd-per-128b-word bulk SDRAM upload via PUSH_W32_AUTO."""
        t0 = time.perf_counter()
        self.queue_cmd(OP_SDRAM_CMD, SDRAM_SUB_SET_ADDR, start_w128 & 0x7FFFF)
        total = len(words_128b)
        for i, w in enumerate(words_128b):
            self.queue_cmd(OP_SDRAM_CMD, SDRAM_SUB_PUSH_AUTO, w & 0xFFFFFFFF)
            self.queue_cmd(OP_SDRAM_CMD, SDRAM_SUB_PUSH_AUTO, (w >> 32) & 0xFFFFFFFF)
            self.queue_cmd(OP_SDRAM_CMD, SDRAM_SUB_PUSH_AUTO, (w >> 64) & 0xFFFFFFFF)
            self.queue_cmd(OP_SDRAM_CMD, SDRAM_SUB_PUSH_AUTO, (w >> 96) & 0xFFFFFFFF)
            if (i & 0x3FF) == 0x3FF:
                self.flush()
                if progress_cb is not None:
                    progress_cb(i + 1, total)
        self.flush()
        self.wait_sdram_idle(timeout_s=10.0)
        return time.perf_counter() - t0

    def sdram_read_128b(self, w128_addr: int) -> int:
        self.queue_cmd(OP_SDRAM_CMD, SDRAM_SUB_RD128 | (w128_addr & 0x7FFFF), 0)
        self.flush()
        self.wait_sdram_idle()
        lo = self.read_resp(0x2006)
        hi = self.read_resp(0x2007)
        return lo | (hi << 64)

    def sdram_run_bist(self, start_w128: int, len_w128: int, timeout_s: float = 10.0) -> dict:
        self.queue_cmd(OP_SDRAM_CMD, SDRAM_SUB_BIST | (start_w128 & 0x7FFFF), len_w128 & 0xFFFFF)
        self.flush()
        self.wait_sdram_idle(timeout_s=timeout_s)
        cyc = self.read_resp(0x2008)
        err_stall = self.read_resp(0x2009)
        return {
            "wr_cyc": cyc & 0xFFFFFFFF,
            "rd_cyc": (cyc >> 32) & 0xFFFFFFFF,
            "err_cnt": err_stall & 0xFFFFFFFF,
            "bytes": len_w128 * 16,
        }

    def wait_done(self, timeout_s: float = 2.0) -> dict:
        t0 = time.monotonic()
        while True:
            st = self.read_resp(0x2000)
            if (st >> 1) & 1:
                break
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"FPGA engine timed out (status=0x{st:016x})")
        c1 = self.read_resp(0x2001)
        c2 = self.read_resp(0x2002)
        c3 = self.read_resp(0x2003)
        return {
            "status": f"0x{st:016x}",
            "total_cycles": c1 & 0xFFFFFFFF,
            "mac_cycles": (c1 >> 32) & 0xFFFFFFFF,
            "epilogue_cycles": c2 & 0xFFFFFFFF,
            "wgt_reads": (c2 >> 32) & 0xFFFFFFFF,
            "out_checksum": f"0x{c3 & 0xFFFFFFFF:08x}",
            "run_count": (c3 >> 32) & 0xFFFFFFFF,
        }


def load_hex_ints(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]


def pack_cfg0(op_mode: int, w4: int, lut_mode: int, dim_t: int, dim_k: int, dim_n: int) -> int:
    return (
        (op_mode & 0x3)
        | ((w4 & 0x1) << 2)
        | ((lut_mode & 0x3) << 3)
        | ((dim_t & 0xFF) << 5)
        | ((dim_k & 0xFF) << 13)
        | ((dim_n & 0xFF) << 21)
    )


def pack_cfg1(dim_n2: int, act_base: int, wgt_base: int, hpost_base: int) -> int:
    return (
        (dim_n2 & 0xFF)
        | ((act_base & 0xFF) << 8)
        | ((wgt_base & 0xFF) << 16)
        | ((hpost_base & 0xFF) << 24)
    )


def pack_cfg2(wgt2_base: int, param1_base: int, param2_base: int, out_base: int) -> int:
    return (
        (wgt2_base & 0xFF)
        | ((param1_base & 0xFF) << 8)
        | ((param2_base & 0xFF) << 16)
        | ((out_base & 0xFF) << 24)
    )


def run_ffn_workload(dev: UsbBlasterJtag, folder: str, w4_mode: int, run_mode_v_check: bool = False) -> dict:
    vdir = GOLDEN_DIR / folder
    act_words = load_hex_ints(vdir / "act_words_128b.hex")
    wgt_words = load_hex_ints(vdir / "wgt_words_128b.hex")
    param_words = load_hex_ints(vdir / "param_words_64b.hex")
    lut_bytes = load_hex_ints(vdir / "swoosh_l_lut.hex")
    hpost_golden = load_hex_ints(vdir / "h_post_words_128b.hex")
    y_golden = load_hex_ints(vdir / "y_golden_words_128b.hex")

    t_up0 = time.perf_counter()
    for i, w in enumerate(act_words):
        dev.queue_write_128b(OP_COMMIT_ACT, i, w)
    for i, w in enumerate(wgt_words):
        dev.queue_write_128b(OP_COMMIT_WGT, i, w)
    for i, p in enumerate(param_words):
        dev.queue_write_64b_param(i, p)
    for i, b in enumerate(lut_bytes):
        dev.queue_cmd(OP_WRITE_LUT, i, b)
    dev.flush()
    upload_ms = (time.perf_counter() - t_up0) * 1e3

    wgt2_base = 64 if w4_mode else 128

    mode_v_info = None
    if run_mode_v_check:
        # Mode V: Run Layer 1 alone -> read back h_post via JTAG -> Run Layer 2 alone -> read back y_out
        t_v0 = time.perf_counter()
        dev.queue_cmd(OP_WRITE_REG, 0, pack_cfg0(0, w4_mode, 1, 16, 32, 64))
        dev.queue_cmd(OP_WRITE_REG, 1, pack_cfg1(32, 0, 0, 64))
        dev.queue_cmd(OP_WRITE_REG, 2, pack_cfg2(wgt2_base, 0, 64, 0))
        dev.queue_cmd(OP_WRITE_REG, 3, 0)
        dev.queue_cmd(OP_START, 0, 0)
        dev.flush()
        st1 = dev.wait_done()
        hpost_v = [dev.read_128b(0, i) for i in range(64)]
        mism_l1 = sum(1 for a, b in zip(hpost_v, hpost_golden) if a != b)

        # Load h_post back to act_mem[64..127] as Host would in Mode V, then run Layer 2
        for i, w in enumerate(hpost_v):
            dev.queue_write_128b(OP_COMMIT_ACT, 64 + i, w)
        dev.queue_cmd(OP_WRITE_REG, 0, pack_cfg0(0, w4_mode, 0, 16, 64, 32))
        dev.queue_cmd(OP_WRITE_REG, 1, pack_cfg1(32, 64, wgt2_base, 64))
        dev.queue_cmd(OP_WRITE_REG, 2, pack_cfg2(wgt2_base, 64, 64, 0))
        dev.queue_cmd(OP_START, 0, 0)
        dev.flush()
        st2 = dev.wait_done()
        yout_v = [dev.read_128b(0, i) for i in range(32)]
        mism_l2 = sum(1 for a, b in zip(yout_v, y_golden) if a != b)
        mode_v_ms = (time.perf_counter() - t_v0) * 1e3
        assert mism_l1 == 0 and mism_l2 == 0, f"Mode V mismatch: L1={mism_l1}, L2={mism_l2}"
        mode_v_info = {
            "host_wall_ms": round(mode_v_ms, 2),
            "l1_cycles": st1["total_cycles"],
            "l2_cycles": st2["total_cycles"],
            "l1_mismatch_words": mism_l1,
            "l2_mismatch_words": mism_l2,
        }

    # Mode S: Chained 2-layer FFN on-chip in single OP_START
    t_s0 = time.perf_counter()
    dev.queue_cmd(OP_WRITE_REG, 0, pack_cfg0(1, w4_mode, 1, 16, 32, 64))
    dev.queue_cmd(OP_WRITE_REG, 1, pack_cfg1(32, 0, 0, 64))
    dev.queue_cmd(OP_WRITE_REG, 2, pack_cfg2(wgt2_base, 0, 64, 0))
    dev.queue_cmd(OP_WRITE_REG, 3, 0)
    dev.queue_cmd(OP_START, 0, 0)
    dev.flush()
    counters = dev.wait_done()
    mode_s_exec_plus_poll_ms = (time.perf_counter() - t_s0) * 1e3

    t_rd0 = time.perf_counter()
    hpost_s = [dev.read_128b(1, 64 + i) for i in range(64)]
    yout_s = [dev.read_128b(0, i) for i in range(32)]
    readback_ms = (time.perf_counter() - t_rd0) * 1e3

    mism_hpost = sum(1 for a, b in zip(hpost_s, hpost_golden) if a != b)
    mism_yout = sum(1 for a, b in zip(yout_s, y_golden) if a != b)
    assert mism_hpost == 0 and mism_yout == 0, f"{folder} Mode S mismatch: hpost={mism_hpost}, yout={mism_yout}"

    mac_util_pct = round(100.0 * counters["mac_cycles"] / counters["total_cycles"], 2)
    return {
        "workload": folder,
        "w4_mode": bool(w4_mode),
        "upload_jtag_ms": round(upload_ms, 2),
        "readback_jtag_ms": round(readback_ms, 2),
        "mode_s_poll_ms": round(mode_s_exec_plus_poll_ms, 2),
        "hpost_mismatch_words": mism_hpost,
        "yout_mismatch_words": mism_yout,
        "zero_lsb_verified": True,
        "counters": counters,
        "mac_lane_utilization_pct": mac_util_pct,
        "mode_v": mode_v_info,
    }


def run_joiner_workload(dev: UsbBlasterJtag) -> dict:
    vdir = GOLDEN_DIR / "joiner_W8A8"
    enc_words = load_hex_ints(vdir / "enc_words_128b.hex")
    dec_words = load_hex_ints(vdir / "dec_words_128b.hex")
    wgt_words = load_hex_ints(vdir / "wgt_words_128b.hex")
    param_words = load_hex_ints(vdir / "param_words_64b.hex")
    tanh_bytes = load_hex_ints(vdir / "tanh_lut.hex")
    j_act_golden = load_hex_ints(vdir / "j_act_frames0_15.hex")
    logit_golden = load_hex_ints(vdir / "logit_golden_words_128b.hex")

    t_up0 = time.perf_counter()
    for i, w in enumerate(enc_words):
        dev.queue_write_128b(OP_COMMIT_ACT, i, w)
    for i, w in enumerate(dec_words):
        dev.queue_write_128b(OP_COMMIT_ACT, 32 + i, w)
    for i, w in enumerate(wgt_words):
        dev.queue_write_128b(OP_COMMIT_WGT, i, w)
    for i, p in enumerate(param_words):
        dev.queue_write_64b_param(i, p)
    for i, b in enumerate(tanh_bytes):
        dev.queue_cmd(OP_WRITE_LUT, 256 + i, b)
    dev.flush()
    upload_ms = (time.perf_counter() - t_up0) * 1e3

    dev.queue_cmd(OP_WRITE_REG, 0, pack_cfg0(2, 0, 2, 16, 32, 32))
    dev.queue_cmd(OP_WRITE_REG, 1, pack_cfg1(64, 0, 0, 0))
    dev.queue_cmd(OP_WRITE_REG, 2, pack_cfg2(0, 0, 0, 0))
    dev.queue_cmd(OP_WRITE_REG, 3, 32)
    dev.queue_cmd(OP_START, 0, 0)
    dev.flush()
    counters = dev.wait_done()

    t_rd0 = time.perf_counter()
    logit_hw = [dev.read_128b(0, i) for i in range(32)]
    readback_ms = (time.perf_counter() - t_rd0) * 1e3

    mism_logit = sum(1 for a, b in zip(logit_hw, logit_golden) if a != b)
    assert mism_logit == 0, f"joiner_W8A8 mismatch: {mism_logit}"

    return {
        "workload": "joiner_W8A8",
        "upload_jtag_ms": round(upload_ms, 2),
        "readback_jtag_ms": round(readback_ms, 2),
        "logit_mismatch_words": mism_logit,
        "zero_lsb_verified": True,
        "counters": counters,
        "j_act_bytes": len(j_act_golden),
    }


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    with UsbBlasterJtag() as dev:
        id_reg = dev.read_resp(0x2000)
        cap_reg = dev.read_resp(0x2005)
        magic = (id_reg >> 32) & 0xFFFFFFFF
        assert magic == 0x5A463445, f"Unexpected FPGA MAGIC 0x{magic:08x}"
        num_lanes = (cap_reg >> 16) & 0xFFFF

        # Echo loopback test
        test_val = 0xA55A1234DEADBEEF
        dev.queue_cmd(OP_STAGE_W32, 0, test_val & 0xFFFFFFFF)
        dev.queue_cmd(OP_STAGE_W32, 1, (test_val >> 32) & 0xFFFFFFFF)
        dev.flush()
        echo_got = dev.read_resp(0x3000)
        assert echo_got == test_val, f"Echo mismatch: got 0x{echo_got:016x}"

        # Measure real board oscillator frequency via free_clk_cnt over 0.25s intervals (3 samples)
        freq_samples_mhz = []
        for _ in range(3):
            c0 = dev.read_resp(0x2004)
            t0 = time.perf_counter()
            time.sleep(0.25)
            c1 = dev.read_resp(0x2004)
            t1 = time.perf_counter()
            freq_samples_mhz.append(round(((c1 - c0) / (t1 - t0)) / 1e6, 3))

        r_ffn_w8 = run_ffn_workload(dev, "stage0_layer0_ffn1_W8A8", w4_mode=0, run_mode_v_check=True)
        r_ffn_w4 = run_ffn_workload(dev, "stage0_layer0_ffn1_W4A8", w4_mode=1, run_mode_v_check=False)
        r_joiner = run_joiner_workload(dev)

    # Calculate real-board execution time at measured 50 MHz and Quartus Fmax 80.34 MHz
    clk_mhz = sum(freq_samples_mhz) / len(freq_samples_mhz)
    fmax_mhz = 80.34
    for r in [r_ffn_w8, r_ffn_w4, r_joiner]:
        cyc = r["counters"]["total_cycles"]
        r["hw_exec_us_at_50mhz"] = round(cyc / clk_mhz, 3)
        r["hw_exec_us_at_fmax_80_34mhz"] = round(cyc / fmax_mhz, 3)

    w8_reads = r_ffn_w8["counters"]["wgt_reads"]
    w4_reads = r_ffn_w4["counters"]["wgt_reads"]

    results = {
        "gate": "G4",
        "device": "EP4CE10E22C8N",
        "magic": f"0x{magic:08x}",
        "num_lanes": num_lanes,
        "echo_loopback_pass": True,
        "measured_board_clk_mhz_samples": freq_samples_mhz,
        "measured_board_clk_mhz_mean": round(clk_mhz, 3),
        "w4_weight_read_reduction_pct": round(100.0 * (1.0 - w4_reads / w8_reads), 2),
        "workloads": {
            "ffn_w8a8": r_ffn_w8,
            "ffn_w4a8": r_ffn_w4,
            "joiner_w8a8": r_joiner,
        },
    }
    out_path = BUILD_DIR / "g4_hardware_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
