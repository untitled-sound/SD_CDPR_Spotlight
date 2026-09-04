"""
cdpr/motor_diagnostic.py
========================
Phase 1 diagnostic: test each motor individually in Extended Position Mode.

Runs a fixed sequence of encoder-count steps on a single motor at a time,
printing the commanded and measured counts at each step so you can:

  1. Confirm the motor responds to commands at all
  2. Confirm the winding direction matches expectation
  3. Confirm counts-per-metre calibration against a physical ruler

Extended Position Mode (operating mode 4) allows encoder counts beyond the
0-4095 range of standard position mode, into approximately ±1,048,575 counts
(~256 full turns in each direction).  This eliminates the clamp problem that
causes motors to stall when delta counts would cross below zero.

Usage
-----
  # Test M1 (ID 1) — winds CW, so positive delta = shorter cable
  python3 cdpr/motor_diagnostic.py --id 1

  # Test M4 (ID 4) — same winding direction as M1
  python3 cdpr/motor_diagnostic.py --id 4

  # Test both in sequence
  python3 cdpr/motor_diagnostic.py --id 1 4

  # Custom step size (default 100 mm)
  python3 cdpr/motor_diagnostic.py --id 1 --step_mm 50

  # Dry run in simulation (no hardware)
  python3 cdpr/motor_diagnostic.py --id 1 4 --sim

What to observe
---------------
  For ID 1 (CW tightens, winding_sign = +1):
    Positive delta counts → motor turns CW → cable shortens (winds in)
    Negative delta counts → motor turns CCW → cable lengthens (pays out)

  For ID 4 (CW tightens, winding_sign = +1):
    Same as ID 1.

  If a motor moves in the WRONG direction:
    Flip its winding_sign in CDPRConfig from +1 to -1.

  If a motor does NOT move at all:
    - Check LED: slow red blink = hardware error, check Dynamixel Wizard
    - Ping confirms connection but no motion usually = wrong operating mode
      or torque not enabled — the script handles both, so check wiring first.

  Measuring calibration with a ruler:
    At step_mm = 100, the spool should advance exactly 100 mm of cable.
    counts_per_metre = 4096 / (2π × 0.100) = 6519 counts/m
    100 mm = 652 counts
    Measure actual cable movement and compare.
"""

from __future__ import annotations
import argparse
import signal
import sys
import time
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("motor_diag")

# ── Constants ──────────────────────────────────────────────────────────────

DEVICE       = "/dev/ttyUSB0"
BAUD_RATE    = 1_000_000          # 1 M baud
PROTOCOL     = 2.0
R_SPOOL      = 0.100              # metres
ENCODER_CPR  = 4096
COUNTS_PER_M = ENCODER_CPR / (2 * 3.14159265 * R_SPOOL)   # ≈ 6519

# Winding signs for M1 and M4 (both CW-tighten)
WINDING_SIGNS = {1: +1, 4: +1}

# XL330 Control Table
ADDR_OP_MODE        = 11
ADDR_TORQUE_ENABLE  = 64
ADDR_PROFILE_VEL    = 112
ADDR_GOAL_POSITION  = 116   # Extended: 4 bytes, signed 32-bit
ADDR_PRESENT_POS    = 132   # Extended: 4 bytes, signed 32-bit

MODE_EXTENDED_POS   = 4     # Extended Position Control — no 0-4095 limit
TORQUE_ON           = 1
TORQUE_OFF          = 0

# Conservative profile velocity for diagnostic (≈ 9 rpm)
DIAG_PROFILE_VEL    = 40    # Dynamixel units × 0.229 rpm


# ── Helper: pack signed 32-bit int to 4 bytes (little-endian) ─────────────

def pack4(value: int) -> list:
    v = int(value) & 0xFFFFFFFF   # two's complement 32-bit
    return [
        v & 0xFF,
        (v >> 8)  & 0xFF,
        (v >> 16) & 0xFF,
        (v >> 24) & 0xFF,
    ]


def unpack4_signed(raw: int) -> int:
    """Convert unsigned 32-bit raw SDK read to signed int."""
    if raw > 0x7FFFFFFF:
        raw -= 0x100000000
    return raw


# ── Single-motor driver ────────────────────────────────────────────────────

class SingleMotorDriver:
    """
    Minimal direct SDK wrapper for one motor.
    Uses Extended Position Mode (mode 4).
    """

    def __init__(self, motor_id: int, port, packet_handler):
        self.id  = motor_id
        self.port = port
        self.ph   = packet_handler
        self.sign = WINDING_SIGNS.get(motor_id, +1)

    def _write1(self, addr, val):
        comm, err = self.ph.write1ByteTxRx(self.port, self.id, addr, val)
        if comm != 0:
            log.warning(f"  ID{self.id} write1 @{addr}: {self.ph.getTxRxResult(comm)}")
        if err != 0:
            log.warning(f"  ID{self.id} hw error @{addr}: {self.ph.getRxPacketError(err)}")
        return comm == 0

    def _write4(self, addr, val):
        from dynamixel_sdk import GroupSyncWrite
        sw = GroupSyncWrite(self.port, self.ph, addr, 4)
        sw.addParam(self.id, pack4(val))
        result = sw.txPacket()
        sw.clearParam()
        if result != 0:
            log.warning(f"  ID{self.id} write4 @{addr}: {self.ph.getTxRxResult(result)}")
        return result == 0

    def _read4_signed(self, addr) -> int:
        raw, comm, err = self.ph.read4ByteTxRx(self.port, self.id, addr)
        if comm != 0:
            log.warning(f"  ID{self.id} read4 @{addr}: {self.ph.getTxRxResult(comm)}")
            return None
        return unpack4_signed(raw)

    def ping(self) -> bool:
        _, comm, _ = self.ph.ping(self.port, self.id)
        return comm == 0

    def setup(self):
        """Disable torque, set extended position mode, set profile vel, enable torque."""
        log.info(f"  ID{self.id}: disabling torque …")
        self._write1(ADDR_TORQUE_ENABLE, TORQUE_OFF)
        time.sleep(0.05)

        log.info(f"  ID{self.id}: setting Extended Position Mode (mode 4) …")
        self._write1(ADDR_OP_MODE, MODE_EXTENDED_POS)
        time.sleep(0.05)

        log.info(f"  ID{self.id}: setting profile velocity = {DIAG_PROFILE_VEL} "
                 f"({DIAG_PROFILE_VEL * 0.229:.1f} rpm) …")
        self._write4(ADDR_PROFILE_VEL, DIAG_PROFILE_VEL)
        time.sleep(0.05)

        log.info(f"  ID{self.id}: enabling torque …")
        self._write1(ADDR_TORQUE_ENABLE, TORQUE_ON)
        time.sleep(0.1)

    def shutdown(self):
        log.info(f"  ID{self.id}: disabling torque (shutdown) …")
        self._write1(ADDR_TORQUE_ENABLE, TORQUE_OFF)

    def read_position(self) -> int | None:
        return self._read4_signed(ADDR_PRESENT_POS)

    def send_goal(self, goal_count: int):
        """Send a goal position in extended counts (signed 32-bit)."""
        self._write4(ADDR_GOAL_POSITION, goal_count)

    def mm_to_delta_counts(self, mm: float) -> int:
        """
        Convert a cable length change in mm to a delta encoder count.

        Positive mm  = cable shortens (winds in).
        winding_sign applies here:
          +1 (CW tightens): winding in  → positive encoder delta
          -1 (CCW tightens): winding in → negative encoder delta
        """
        delta_counts = (mm / 1000.0) * COUNTS_PER_M * self.sign
        return int(round(delta_counts))


# ── Simulated driver (no hardware) ────────────────────────────────────────

class SimulatedMotorDriver:
    """Fake driver that echoes commands with a small lag."""

    def __init__(self, motor_id: int):
        self.id   = motor_id
        self.sign = WINDING_SIGNS.get(motor_id, +1)
        self._pos = 0
        self._goal = 0

    def ping(self) -> bool:
        return True

    def setup(self):
        log.info(f"  [SIM] ID{self.id}: setup OK")

    def shutdown(self):
        log.info(f"  [SIM] ID{self.id}: shutdown")

    def read_position(self) -> int:
        # Converge toward goal
        self._pos += int((self._goal - self._pos) * 0.7)
        return self._pos

    def send_goal(self, goal_count: int):
        self._goal = goal_count

    def mm_to_delta_counts(self, mm: float) -> int:
        delta_counts = (mm / 1000.0) * COUNTS_PER_M * self.sign
        return int(round(delta_counts))


# ── Diagnostic sequence ────────────────────────────────────────────────────

STEP_SEQUENCE = [
    # (description,          cable_change_mm)
    ("Wind in   +100 mm",    +100),
    ("Pay out   -100 mm",    -100),
    ("Wind in   +200 mm",    +200),
    ("Pay out   -200 mm",    -200),
    ("Return to home (0)",      0),   # special: go to absolute zero
]


def run_diagnostic(driver, step_mm: float, settle_s: float):
    """
    Execute the diagnostic step sequence on one motor.

    For each step:
      - Print expected direction and count delta
      - Command the motor
      - Wait for it to settle
      - Read back actual position
      - Print comparison
    """

    log.info(f"\n{'─'*60}")
    log.info(f"  Diagnostic — Motor ID {driver.id}  "
             f"(winding sign = {driver.sign:+d})")
    log.info(f"  step_mm={step_mm}  settle={settle_s}s")
    log.info(f"  counts_per_metre ≈ {COUNTS_PER_M:.1f}")
    log.info(f"{'─'*60}")

    # Record home position (current counts = zero reference)
    home_counts = driver.read_position()
    if home_counts is None:
        log.error(f"  Cannot read position from ID{driver.id}. Skipping.")
        return

    log.info(f"  Home encoder position: {home_counts} counts\n")
    print(f"  {'Step':<24} {'Goal Δcts':>10} {'Goal abs':>10} "
          f"{'Read abs':>10} {'Δ err':>8}  {'Cable mm':>10}")
    print(f"  {'─'*24} {'─'*10} {'─'*10} {'─'*10} {'─'*8}  {'─'*10}")

    prev_goal_abs = home_counts

    for desc, cable_mm in STEP_SEQUENCE:
        if cable_mm == 0:
            # Return to home
            goal_abs = home_counts
        else:
            delta_cts = driver.mm_to_delta_counts(step_mm
                        if cable_mm > 0 else -step_mm) * (1 if cable_mm > 0 else -1)
            # recompute with the actual sign/magnitude of this step
            delta_cts = driver.mm_to_delta_counts(cable_mm)
            goal_abs  = home_counts + delta_cts

        delta_from_home = goal_abs - home_counts
        cable_change_mm = (-delta_from_home / COUNTS_PER_M / driver.sign) * 1000.0

        driver.send_goal(goal_abs)
        time.sleep(settle_s)

        actual_abs = driver.read_position()
        if actual_abs is None:
            actual_abs = -9999

        err = actual_abs - goal_abs

        print(f"  {desc:<24} {delta_from_home:>+10d} {goal_abs:>10d} "
              f"{actual_abs:>10d} {err:>+8d}  {cable_change_mm:>+9.1f}mm")

        prev_goal_abs = goal_abs

    log.info(f"\n  Diagnostic complete for ID{driver.id}.")
    log.info(f"  Check the 'Δ err' column — values near 0 = motor arrived.")
    log.info(f"  Check cable direction against 'Cable mm':")
    log.info(f"    Positive mm = cable shortened (wound in)")
    log.info(f"    Negative mm = cable lengthened (paid out)")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Motor diagnostic — Extended Position Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--id",      type=int, nargs="+", default=[1, 4],
                        help="Motor ID(s) to test (e.g. --id 1 4)")
    parser.add_argument("--step_mm", type=float, default=100.0,
                        help="Cable step size in mm for each move")
    parser.add_argument("--settle",  type=float, default=2.0,
                        help="Seconds to wait after each command")
    parser.add_argument("--device",  default=DEVICE)
    parser.add_argument("--baud",    type=int, default=BAUD_RATE)
    parser.add_argument("--sim",     action="store_true",
                        help="Simulate motors — no hardware needed")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    drivers = []

    if args.sim:
        log.info("=== SIMULATION MODE — no hardware ===")
        for mid in args.id:
            drivers.append(SimulatedMotorDriver(mid))
    else:
        try:
            from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
        except ImportError:
            log.error("dynamixel_sdk not found. Install with: pip install dynamixel-sdk")
            sys.exit(1)

        port = PortHandler(args.device)
        ph   = PacketHandler(PROTOCOL)

        if not port.openPort():
            log.error(f"Cannot open port {args.device}")
            sys.exit(1)
        log.info(f"Port opened: {args.device}")

        if not port.setBaudRate(args.baud):
            log.error(f"Cannot set baud rate {args.baud}")
            sys.exit(1)
        log.info(f"Baud rate set: {args.baud}")

        for mid in args.id:
            d = SingleMotorDriver(mid, port, ph)
            log.info(f"Pinging ID{mid} …")
            if not d.ping():
                log.error(f"  ID{mid} not responding. Check wiring, ID, and baud rate.")
                sys.exit(1)
            log.info(f"  ID{mid} OK")
            drivers.append(d)

    # ── Shutdown handler ───────────────────────────────────────────────────
    def _shutdown(sig=None, frame=None):
        print()
        log.info("Interrupt — disabling torque on all motors …")
        for d in drivers:
            try:
                d.shutdown()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Setup all motors ───────────────────────────────────────────────────
    log.info("\nSetting up motors …")
    for d in drivers:
        d.setup()

    # ── Run diagnostics ────────────────────────────────────────────────────
    for d in drivers:
        input(f"\n  Press ENTER to begin diagnostic on ID{d.id} …\n")
        run_diagnostic(d, args.step_mm, args.settle)
        time.sleep(0.5)

    # ── Shutdown ───────────────────────────────────────────────────────────
    log.info("\nAll diagnostics complete. Disabling torque …")
    for d in drivers:
        d.shutdown()

    if not args.sim:
        port.closePort()
        log.info("Port closed.")


if __name__ == "__main__":
    main()