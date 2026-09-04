import sys
import time
import math
import matplotlib.pyplot as plt
from dynamixel_sdk import (
    PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS
)

ADDR_PRESENT_CURRENT = 126   # 2 bytes
SUPPLY_VOLTAGE = 5.0       # adjust if needed

DEVICE   = "/dev/ttyUSB0"
BAUDRATE = 1_000_000
PROTOCOL = 2.0

R_SPOOL        = 0.050
COUNTS_PER_REV = 4096
COUNTS_PER_M   = COUNTS_PER_REV / (2 * math.pi * R_SPOOL)  #13038 counts/m

Z_EE = 0.05     # EE height 

# Anchors
ANCHORS = {
    1: [0.00, 0.00, 2.00],   # M1 back-left
    2: [0.00, 1.48, 2.00],   # M2 front-left
    3: [1.43, 1.48, 2.00],   # M3 front-right
    4: [1.43, 0.00, 2.00],   # M4 back-right
}

# centre
HOME = [1.43 / 2, 1.48 / 2, Z_EE]   # [0.715, 0.740, 1.00]
SIGNS       = {1: +1, 2: -1, 3: +1, 4: -1}

MOTOR_IDS = [1, 2, 3, 4]

# Motor settings
PROFILE_VEL = 20    # velocity units × 0.229 rpm = 9 rpm 
SETTLE_S    = 17   # wait time

# Control table addresses
ADDR_OP_MODE  = 11
ADDR_TORQUE   = 64
ADDR_PROF_VEL = 112
ADDR_GOAL_POS = 116   
ADDR_PRES_POS = 132   

def read_current(port, ph, mid):
    raw, _, _ = ph.read2ByteTxRx(port, mid, ADDR_PRESENT_CURRENT)
    return raw - 0x10000 if raw > 0x7FFF else raw

def cable_length(p, anchor):
    return math.sqrt(sum((p[k] - anchor[k])**2 for k in range(3)))

def ik(p):
    return {mid: cable_length(p, ANCHORS[mid]) for mid in MOTOR_IDS}

def delta_counts(L_home, L_target):

    deltas = {}
    for mid in MOTOR_IDS:
        dL = L_target[mid] - L_home[mid]
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

def move_to(port, ph, home_counts, L_home, target_xyz, t_log, p_log, t0_global):
    L_tgt   = ik(target_xyz)
    deltas  = delta_counts(L_home, L_tgt)
    goals   = {mid: home_counts[mid] + deltas[mid] for mid in MOTOR_IDS}

    sync_write_goals(port, ph, goals)
    print(f"\n  Moving")
    time.sleep(SETTLE_S)

    t_start = time.time()

    while time.time() - t_start < SETTLE_S:
        t = time.time() - t0_global
        t_log.append(t)

        for mid in MOTOR_IDS:
            i = read_current(port, ph, mid)
            p_log[mid].append(i * SUPPLY_VOLTAGE)

        time.sleep(0.05)


def main():
    t_log = []
    p_log = {mid: [] for mid in MOTOR_IDS}
    t0_global = time.time()

    target = [HOME[0], HOME[1], HOME[2]+1.5]

    # Basic bounds check
    if not (0 < target[0] < 1.43 and 0 < target[1] < 1.48):
        sys.exit("out of bounds")

    print(f"\nCounts per metre : {COUNTS_PER_M:.1f}")
    print(f"mm per count     : {1000/COUNTS_PER_M:.4f}")

    port = PortHandler(DEVICE)
    ph   = PacketHandler(PROTOCOL)

    if not port.openPort():
        sys.exit("Cannot open port.")
    if not port.setBaudRate(BAUDRATE):
        sys.exit("Cannot set baud rate.")

    for mid in MOTOR_IDS:
        ph.write1ByteTxRx(port, mid, ADDR_TORQUE,   0)
        ph.write1ByteTxRx(port, mid, ADDR_OP_MODE,  4)   # extended position
        ph.write4ByteTxRx(port, mid, ADDR_PROF_VEL, PROFILE_VEL)
        ph.write1ByteTxRx(port, mid, ADDR_TORQUE,   1)

    home_counts = {mid: read_pos(port, ph, mid) for mid in MOTOR_IDS}
    L_home      = ik(HOME)

    print(f"\nHome counts : { {mid: home_counts[mid] for mid in MOTOR_IDS} }")
    print(f"Home lengths: { {mid: round(L_home[mid],4) for mid in MOTOR_IDS} }")

    move_to(port, ph, home_counts, L_home, target, t_log, p_log, t0_global)
    target = [HOME[0], HOME[1], HOME[2]+1.0]
    SETTLE_S    = 5   # wait time

    move_to(port, ph, home_counts, L_home, target, t_log, p_log, t0_global)
    target = [HOME[0]+0.35, HOME[1], HOME[2]+1.0]

    move_to(port, ph, home_counts, L_home, target, t_log, p_log, t0_global)
    target = [HOME[0]-0.35, HOME[1], HOME[2]+1.0]

    move_to(port, ph, home_counts, L_home, target, t_log, p_log, t0_global)
    target = [HOME[0], HOME[1]+0.35, HOME[2]+1.0]

    move_to(port, ph, home_counts, L_home, target, t_log, p_log, t0_global)
    target = [HOME[0], HOME[1]-0.35, HOME[2]+1.0]

    move_to(port, ph, home_counts, L_home, target, t_log, p_log, t0_global)
    SETTLE_S    = 10   # wait time


    move_to(port, ph, home_counts, L_home, HOME,   t_log, p_log, t0_global)

    for mid in MOTOR_IDS:
        plt.plot(t_log, p_log[mid], label=f"M{mid}")

    plt.xlabel("Time (s)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid()
    plt.show()

    # for mid in MOTOR_IDS:
    #     ph.write1ByteTxRx(port, mid, ADDR_TORQUE, 0)

    port.closePort()
    print("\nDone.")


if __name__ == "__main__":
    main()