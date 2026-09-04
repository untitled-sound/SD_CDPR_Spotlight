# Li = ||p + R(theta)*bi - ai||

import sys
import time
import numpy as np
from dynamixel_sdk import (
    PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS
)

DEVICE   = "/dev/ttyUSB0"
BAUDRATE = 1_000_000
PROTOCOL = 2.0

R_SPOOL      = 0.050
COUNTS_PER_M = 4096 / (2 * np.pi * R_SPOOL)

Z_EE = 0.00

ANCHORS = np.array([
    [0.00, 0.00, 2.00],
    [0.00, 1.48, 2.00],
    [1.43, 1.48, 2.00],
    [1.43, 0.00, 2.00],
])

B_LOCAL = np.array([
    [-0.070, -0.070, 0.0],
    [-0.070, +0.070, 0.0],
    [+0.070, +0.070, 0.0],
    [+0.070, -0.070, 0.0],
])

# B_LOCAL = np.zeros((4,3))

HOME = np.array([1.43 / 2, 1.48 / 2, Z_EE])

SIGNS       = {1: +1, 2: -1, 3: +1, 4: -1}
MOTOR_IDS   = [1, 2, 3, 4]
PROFILE_VEL = 30
SETTLE_S    = 10.0

ADDR_OP_MODE  = 11
ADDR_TORQUE   = 64
ADDR_PROF_VEL = 112
ADDR_GOAL_POS = 116
ADDR_PRES_POS = 132


def R(phi, gam):
    cp, sp = np.cos(phi), np.sin(phi)
    cg, sg = np.cos(gam), np.sin(gam)
    Ry = np.array([[ cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cg, -sg], [0, sg, cg]])
    return Ry @ Rx
 
def ik(p, phi=0.0, gam=0.0):
    Rm = R(phi, gam)
    return [float(np.linalg.norm(p + Rm @ B_LOCAL[i] - ANCHORS[i])) for i in range(4)]

def fk(L_meas, p0, phi0=0.0, gam0=0.0, iters=200, tol=1e-9, eps=1e-6):
    q = np.array([p0[0], p0[1], phi0, gam0])
    prev = 1e9
    for _ in range(iters):
        x, y, phi, gam = q
        p = np.array([x, y, Z_EE])
        Lp = np.array(ik(p, phi, gam))
        r  = Lp - L_meas
        cur = float(np.linalg.norm(r))
        if cur > prev * 2: break
        prev = cur
        J = np.zeros((4, 4))
        for j in range(4):
            qe = q.copy(); qe[j] += eps
            pe = np.array([qe[0], qe[1], Z_EE])
            J[:, j] = (np.array(ik(pe, qe[2], qe[3])) - Lp) / eps
        try: dq = -np.linalg.solve(J.T @ J, J.T @ r)
        except: break
        mx = max(max(abs(dq[:2])), 1e-9); ma = max(max(abs(dq[2:])), 1e-9)
        dq *= min(1.0, 0.005 / mx, 0.05 / ma)
        q  += dq
        if np.linalg.norm(dq) < tol: break
    x, y, phi, gam = q
    return x, y, phi, gam
 
 
def delta_counts(L_home, L_target):
    deltas = {}
    for i, mid in enumerate(MOTOR_IDS):
        dL = L_target[i] - L_home[i]
        deltas[mid] = int(-dL * COUNTS_PER_M * SIGNS[mid])
    return deltas


def pack4(value):
    v = int(value) & 0xFFFFFFFF
    return [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF]
 
def read_pos(port, ph, mid):
    raw, _, _ = ph.read4ByteTxRx(port, mid, ADDR_PRES_POS)
    return raw - 0x100000000 if raw > 0x7FFFFFFF else raw
 
def sync_write_goals(port, ph, goal_counts):
    sw = GroupSyncWrite(port, ph, ADDR_GOAL_POS, 4)
    for mid in MOTOR_IDS:
        sw.addParam(mid, pack4(goal_counts[mid]))
    sw.txPacket()
    sw.clearParam()
 
 
def move_to(port, ph, home_counts, L_home, target_xyz, phi, gam):
    p = np.array(target_xyz)
    L_tgt  = ik(p, phi, gam)
    deltas = delta_counts(L_home, L_tgt)
    goals  = {mid: home_counts[mid] + deltas[mid] for mid in MOTOR_IDS}
    sync_write_goals(port, ph, goals)
    time.sleep(SETTLE_S)
    counts = {mid: read_pos(port, ph, mid) for mid in MOTOR_IDS}
    L_act  = [L_home[i] + (-( counts[mid] - home_counts[mid]) / (COUNTS_PER_M * SIGNS[mid]))
              for i, mid in enumerate(MOTOR_IDS)]
    phi_e, gam_e = fk(np.array(L_act), p0=p, phi0=phi, gam0=gam)[2:4]
    print(f"  pitch={np.degrees(phi_e):+.2f}°  roll={np.degrees(gam_e):+.2f}°")
    return phi_e, gam_e


def main():
    # if len(sys.argv) == 3:
    #     tx, ty = float(sys.argv[1]), float(sys.argv[2])
    # else:
    #     tx = float(input("Target x [m]: "))
    #     ty = float(input("Target y [m]: "))
 
    # if not (0 < tx < 1.43 and 0 < ty < 1.48):
    #     sys.exit("Out of bounds.")
 
    # Accept actual physical home position — defaults to frame centre.
    # If the EE was positioned by a previous script or by hand at a
    # different location, enter those coordinates so L_home is consistent
    # with the encoder counts the motors are currently holding.
    try:
        hx = float(input(f"Actual home x [m] (Enter = {HOME[0]}): ") or HOME[0])
        hy = float(input(f"Actual home y [m] (Enter = {HOME[1]}): ") or HOME[1])
        hz = float(input(f"Actual home z [m] (Enter = {Z_EE}): ")    or Z_EE)
    except ValueError:
        hx, hy, hz = HOME[0], HOME[1], Z_EE
    true_home = np.array([hx, hy, hz])

    tx, ty = true_home[0] + 0.5, true_home[1]
 
    port = PortHandler(DEVICE)
    ph   = PacketHandler(PROTOCOL)
    if not port.openPort():    sys.exit("Cannot open port.")
    if not port.setBaudRate(BAUDRATE): sys.exit("Cannot set baud rate.")
 
    for mid in MOTOR_IDS:
        ph.write1ByteTxRx(port, mid, ADDR_TORQUE,   0)
        ph.write1ByteTxRx(port, mid, ADDR_OP_MODE,  4)
        ph.write4ByteTxRx(port, mid, ADDR_PROF_VEL, PROFILE_VEL)
        ph.write1ByteTxRx(port, mid, ADDR_TORQUE,   1)
 
    home_counts = {mid: read_pos(port, ph, mid) for mid in MOTOR_IDS}
 
    # L_home derived from actual physical home position, not assumed centre.
    # This makes the IK delta consistent with whatever cable lengths the
    # motors are currently holding, eliminating the L_home mismatch.
    L_home = ik(true_home)
 
    phi, gam = move_to(port, ph, home_counts, L_home, [tx, ty, hz], 0.0, 0.0)
    move_to(port, ph, home_counts, L_home, list(true_home), phi, gam)
 
    for mid in MOTOR_IDS:
        ph.write1ByteTxRx(port, mid, ADDR_TORQUE, 0)
    port.closePort()
 
 
if __name__ == "__main__":
    main()
