"""
cdpr/hardware_test.py
=====================
First physical run script for the 4-cable CDPR.

Executes a 5-point cross-pattern waypoint mission on real hardware,
printing live estimated X/Y position to the terminal every control step
so you can visually compare encoder-FK estimates against physical EE
movement.  All data is logged to CSV for post-run analysis.

Usage
-----
    # From the project root (directory containing cdpr/ and dynamixel_xl330_sync.py):
    python3 cdpr/hardware_test.py

    # Custom home position (if EE is not at frame centre):
    python3 cdpr/hardware_test.py --home_x 0.715 --home_y 0.740

    # Slower speed (recommended for very first run):
    python3 cdpr/hardware_test.py --speed 0.05

    # Dry-run in simulation to verify the mission before touching hardware:
    python3 cdpr/hardware_test.py --sim

Safety checklist before running on hardware
-------------------------------------------
  [ ] End effector physically positioned at the home location
  [ ] All 4 cables hand-tensioned — no slack visible
  [ ] Workspace clear of obstacles
  [ ] Emergency stop: press Ctrl+C at any time — torque disables immediately
  [ ] USB cable to U2D2 secure, motors powered
"""

from __future__ import annotations
import argparse
import logging
import os
import signal
import sys
import time
import csv
from datetime import datetime
import numpy as np

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hardware_test")


# ── Mission definition ────────────────────────────────────────────────────
# 5-point cross pattern.  All coordinates in metres from frame origin.
# Scaled to 60% of the safe workspace to stay well clear of boundary
# on the first run.  Tighten margins once motion is confirmed correct.

def build_mission(cfg) -> list:
    """
    Return waypoints as a list of (x, y) tuples.
    Centre + four cardinal points, then return to centre.
    Each point is 40% of the half-span from centre → conservative first run.
    """
    cx = cfg.Lx / 2.0
    cy = cfg.Ly / 2.0
    rx = cfg.Lx * 0.40 / 2.0   # 40% of half-width  ≈ 0.286 m
    ry = cfg.Ly * 0.40 / 2.0   # 40% of half-depth  ≈ 0.296 m

    return [
        (cx,      cy     ),   # 0 — centre (start)
        (cx,      cy + ry),   # 1 — north
        (cx + rx, cy     ),   # 2 — east
        (cx,      cy - ry),   # 3 — south
        (cx - rx, cy     ),   # 4 — west
        (cx,      cy     ),   # 5 — return to centre
    ]


# ── Live print formatter ──────────────────────────────────────────────────

class LiveDisplay:
    """
    Prints a single overwriting terminal line showing:
      step | ref (x,y) | est (x,y) | err mm | FK residual
    and a separate section header when a new waypoint begins.
    """

    HEADER = (
        "  step  │  ref_x    ref_y  │  est_x    est_y  │  err mm  │  FK res"
    )
    SEP = "─" * 70

    def __init__(self, verbose: bool = False):
        self._step = 0
        self._verbose = verbose
        self._last_wp = -1
        print(self.SEP)
        print(self.HEADER)
        print(self.SEP)

    def update(
        self,
        ref_xy: np.ndarray,
        est_xy: np.ndarray,
        fk_residual: float,
        waypoint_idx: int,
        waypoint_total: int,
    ):
        self._step += 1

        if waypoint_idx != self._last_wp:
            self._last_wp = waypoint_idx
            print()
            log.info(
                f"── Waypoint {waypoint_idx + 1}/{waypoint_total}: "
                f"target ({ref_xy[0]:.3f}, {ref_xy[1]:.3f}) ──"
            )

        err_mm = np.linalg.norm(est_xy - ref_xy) * 1000.0
        line = (
            f"\r  {self._step:5d}  │"
            f"  {ref_xy[0]:6.3f}  {ref_xy[1]:6.3f}  │"
            f"  {est_xy[0]:6.3f}  {est_xy[1]:6.3f}  │"
            f"  {err_mm:6.1f}   │"
            f"  {fk_residual:.2e}"
        )
        print(line, end="", flush=True)

    def newline(self):
        print()


# ── Hardware test runner ──────────────────────────────────────────────────

class HardwareTestRunner:
    """
    Wraps the CDPRController with live display and per-step CSV logging.
    The controller's internal data_log is not used here — we write directly
    so that data is flushed to disk every step (survives Ctrl+C cleanly).
    """

    # Conservative defaults for first hardware run
    LOOP_HZ      = 50.0
    PROFILE_VEL  = 80      # Dynamixel profile velocity units (0.229 rpm each)
                           # 80 ≈ 18.3 rpm — slow and safe for first run

    def __init__(
        self,
        cfg,
        motor_interface,
        home_xy: tuple,
        mission: list,
        speed: float,
        K_t: float,
        T_min: float,
        log_path: str,
        verbose: bool,
    ):
        from cdpr.controller import CDPRController, PIDGains
        from cdpr.kinematics import InverseKinematics, ForwardKinematics

        self.cfg     = cfg
        self.motor   = motor_interface
        self.mission = mission
        self.speed   = speed
        self.home_xy = home_xy
        self.log_path = log_path

        self.ik = InverseKinematics(cfg)
        self.fk = ForwardKinematics(cfg)

        self.ctrl = CDPRController(
            cfg=cfg,
            motor_interface=motor_interface,
            mode='position',
            pid_gains=PIDGains(kp=2.0, ki=0.05, kd=0.10),
            loop_rate_hz=self.LOOP_HZ,
            K_t=K_t,
            T_min=T_min,
        )
        self.ctrl.DEFAULT_PROFILE_VEL = self.PROFILE_VEL

        self.display  = LiveDisplay(verbose=verbose)
        self._running = True
        self._csv_file   = None
        self._csv_writer = None

        # Register shutdown handler
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ── CSV ────────────────────────────────────────────────────────────────

    def _open_csv(self):
        self._csv_file = open(self.log_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'timestamp_s', 'waypoint_idx',
            'ref_x_m', 'ref_y_m',
            'est_x_m', 'est_y_m',
            'err_mm',
            'fk_residual',
            'enc_m1', 'enc_m2', 'enc_m3', 'enc_m4',
            'cable_L1_m', 'cable_L2_m', 'cable_L3_m', 'cable_L4_m',
        ])
        self._csv_file.flush()

    def _write_row(self, t, wp_idx, ref_xy, est_xy, fk_res, counts, lengths):
        err_mm = np.linalg.norm(est_xy - ref_xy) * 1000.0
        self._csv_writer.writerow([
            f"{t:.4f}", wp_idx,
            f"{ref_xy[0]:.5f}", f"{ref_xy[1]:.5f}",
            f"{est_xy[0]:.5f}", f"{est_xy[1]:.5f}",
            f"{err_mm:.2f}",
            f"{fk_res:.4e}",
            counts[0], counts[1], counts[2], counts[3],
            f"{lengths[0]:.5f}", f"{lengths[1]:.5f}",
            f"{lengths[2]:.5f}", f"{lengths[3]:.5f}",
        ])
        self._csv_file.flush()   # flush every row — survives Ctrl+C

    # ── Shutdown ───────────────────────────────────────────────────────────

    def _shutdown(self, sig=None, frame=None):
        self._running = False
        self.display.newline()
        log.info("Shutdown signal received — disabling torque …")
        try:
            self.motor.disable_torque()
        except Exception:
            pass
        if self._csv_file:
            self._csv_file.close()
        log.info(f"Data saved to {self.log_path}")
        sys.exit(0)

    # ── Main run ───────────────────────────────────────────────────────────

    def run(self):
        self._open_csv()
        t_start = time.time()

        log.info("═" * 60)
        log.info("  CDPR HARDWARE TEST — 5-POINT CROSS PATTERN")
        log.info("═" * 60)
        log.info(f"  Frame:    {self.cfg.Lx} × {self.cfg.Ly} × {self.cfg.Hz} m")
        log.info(f"  Spool:    r = {self.cfg.r_spool} m")
        log.info(f"  Winding:  {self.cfg.winding_signs}  (M1,M2,M3,M4)")
        log.info(f"  Speed:    {self.speed} m/s")
        log.info(f"  Profile:  {self.PROFILE_VEL} vel units ≈ "
                 f"{self.PROFILE_VEL * 0.229:.1f} rpm")
        log.info(f"  Home:     ({self.home_xy[0]:.3f}, {self.home_xy[1]:.3f}) m")
        log.info(f"  Log:      {self.log_path}")
        log.info("  Press Ctrl+C at any time to stop safely.")
        log.info("═" * 60)

        # Print mission summary
        log.info("Mission waypoints:")
        for i, (x, y) in enumerate(self.mission):
            log.info(f"  WP {i}: ({x:.3f}, {y:.3f}) m")

        input("\n  Press ENTER when EE is at home position and cables are taut …\n")

        # ── Home ──────────────────────────────────────────────────────────
        log.info("Homing …")
        self.ctrl.home(self.home_xy[0], self.home_xy[1])
        zero_lengths = self.ctrl._zero_lengths
        zero_counts  = self.ctrl._zero_counts
        log.info(f"Zero cable lengths (m): {np.round(zero_lengths, 4)}")
        log.info(f"Zero encoder counts:    {zero_counts.astype(int)}")
        time.sleep(0.5)

        # ── Execute waypoints with live display ───────────────────────────
        from cdpr.kinematics import generate_waypoint_trajectory
        dt = 1.0 / self.LOOP_HZ

        current_xy = np.array(self.home_xy, dtype=float)

        for wp_idx, (wx, wy) in enumerate(self.mission):
            if not self._running:
                break

            log.info(f"Moving to WP {wp_idx}: ({wx:.3f}, {wy:.3f})")

            # Build min-jerk trajectory for this segment
            traj = generate_waypoint_trajectory(
                [tuple(current_xy), (wx, wy)],
                speeds=self.speed,
                dt=dt,
            )

            deadline = time.time() + 30.0   # 30 s per waypoint hard timeout

            for ref_xy in traj:
                if not self._running:
                    break
                if time.time() > deadline:
                    log.warning(f"WP {wp_idx} timeout — skipping to next")
                    break

                t0 = time.time()

                # ── Control step ──────────────────────────────────────────
                goal_lengths, _ = self.ik.cable_lengths(ref_xy[0], ref_xy[1])
                goal_counts = (
                    zero_counts
                    + self.ik.to_encoder_counts(goal_lengths, zero_lengths)
                ).astype(np.int32)
                goal_counts = np.clip(goal_counts, 0, 4095)

                self.motor.send_position_goals(
                    goal_counts,
                    [self.PROFILE_VEL] * 4,
                )

                # ── Read back and estimate position ───────────────────────
                counts = self.motor.read_positions()

                # Recover cable lengths from encoder counts (sign-aware)
                delta_L = (
                    -(counts.astype(float) - zero_counts)
                    / self.cfg.counts_per_metre
                ) * self.cfg.winding_signs
                meas_lengths = zero_lengths + delta_L

                # FK solve
                x_est, y_est, fk_res = self.fk.solve(
                    meas_lengths,
                    x0=current_xy[0],
                    y0=current_xy[1],
                )
                est_xy = np.array([x_est, y_est])
                current_xy = est_xy

                # ── Live display ──────────────────────────────────────────
                self.display.update(
                    ref_xy, est_xy, fk_res,
                    waypoint_idx=wp_idx,
                    waypoint_total=len(self.mission),
                )

                # ── CSV row ───────────────────────────────────────────────
                self._write_row(
                    t=time.time() - t_start,
                    wp_idx=wp_idx,
                    ref_xy=ref_xy,
                    est_xy=est_xy,
                    fk_res=fk_res,
                    counts=counts,
                    lengths=meas_lengths,
                )

                # ── Pace loop ─────────────────────────────────────────────
                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))

            # Arrived at waypoint — brief settle pause
            self.display.newline()
            log.info(
                f"WP {wp_idx} complete. "
                f"Est pos: ({current_xy[0]:.3f}, {current_xy[1]:.3f})  "
                f"Target: ({wx:.3f}, {wy:.3f})  "
                f"Err: {np.linalg.norm(current_xy - np.array([wx,wy]))*1000:.1f} mm"
            )
            time.sleep(0.4)

        # ── Mission complete ───────────────────────────────────────────────
        log.info("═" * 60)
        log.info("  Mission complete.")
        total_t = time.time() - t_start
        log.info(f"  Total time:  {total_t:.1f} s")
        log.info(f"  Data log:    {self.log_path}")
        log.info("═" * 60)

        self.motor.disable_torque()
        if self._csv_file:
            self._csv_file.close()


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CDPR Hardware Test — 5-point cross pattern",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sim",     action="store_true",
                        help="Use simulated motors (no hardware needed)")
    parser.add_argument("--device",  default="/dev/ttyUSB0",
                        help="Serial device path")
    parser.add_argument("--baud",    type=int, default=57600,
                        help="Baud rate")
    parser.add_argument("--home_x",  type=float, default=None,
                        help="Home X position [m] (default=frame centre)")
    parser.add_argument("--home_y",  type=float, default=None,
                        help="Home Y position [m] (default=frame centre)")
    parser.add_argument("--speed",   type=float, default=0.06,
                        help="Travel speed [m/s] — keep low on first run")
    parser.add_argument("--Lx",      type=float, default=1.43)
    parser.add_argument("--Ly",      type=float, default=1.48)
    parser.add_argument("--Hz",      type=float, default=2.00)
    parser.add_argument("--z_ee",    type=float, default=1.00,
                        help="EE operating height [m]")
    parser.add_argument("--K_t",     type=float, default=0.146,
                        help="Motor torque constant [N·m/A] (measured 0.131–0.162)")
    parser.add_argument("--T_min",   type=float, default=0.3,
                        help="Minimum cable tension [N]")
    parser.add_argument("--log",     default=None,
                        help="CSV log file path (default: auto timestamped)")
    parser.add_argument("--verbose", action="store_true",
                        help="Extra debug output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Config ────────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cdpr.kinematics import CDPRConfig
    from cdpr.controller  import MotorInterface, SimulatedMotorInterface

    cfg = CDPRConfig(Lx=args.Lx, Ly=args.Ly, Hz=args.Hz, z_ee=args.z_ee)
    home_x = args.home_x if args.home_x is not None else cfg.Lx / 2
    home_y = args.home_y if args.home_y is not None else cfg.Ly / 2

    # ── Motor interface ───────────────────────────────────────────────────
    if args.sim:
        motor = SimulatedMotorInterface(bandwidth=12.0)
        log.info("Simulation mode — no hardware connection")
    else:
        try:
            from dynamixel_xl330_sync import DynamixelController
        except ImportError:
            log.error(
                "dynamixel_xl330_sync.py not found. "
                "Run from the project root, or use --sim to test without hardware."
            )
            sys.exit(1)

        dxl = DynamixelController(device=args.device, baud=args.baud)
        dxl.connect()
        motor = MotorInterface(dxl)

    # ── Mission ───────────────────────────────────────────────────────────
    mission = build_mission(cfg)

    # ── Log path ──────────────────────────────────────────────────────────
    log_path = args.log or (
        f"cdpr_hw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    # ── Run ───────────────────────────────────────────────────────────────
    runner = HardwareTestRunner(
        cfg=cfg,
        motor_interface=motor,
        home_xy=(home_x, home_y),
        mission=mission,
        speed=args.speed,
        K_t=args.K_t,
        T_min=args.T_min,
        log_path=log_path,
        verbose=args.verbose,
    )
    runner.run()


if __name__ == "__main__":
    main()