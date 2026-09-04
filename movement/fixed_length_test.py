from dynamixel_sdk import *

DEVICE      = "/dev/ttyUSB0"
BAUDRATE    = 1000000
PROTOCOL    = 2.0
MOTOR_IDS   = [1, 2, 3, 4]

R_SPOOL     = 0.050                          # 50 mm radius
COUNTS_PER_REV = 4096
COUNTS_PER_M   = COUNTS_PER_REV / (2 * 3.14159265 * R_SPOOL)  # ≈ 13_038 counts/m

CABLE_MM    = 100                            # how much cable to dispense [mm]
DELTA       = int((CABLE_MM / 1000) * COUNTS_PER_M)  # counts per motor

# Winding signs: +1 = CW tightens (M1, M4),  -1 = CCW tightens (M2, M3)
SIGNS       = {1: +1, 2: -1, 3: +1, 4: -1}

ADDR_OP_MODE  = 11
ADDR_TORQUE   = 64
ADDR_PROF_VEL = 112
ADDR_GOAL_POS = 116
ADDR_PRES_POS = 132

port = PortHandler(DEVICE)
ph   = PacketHandler(PROTOCOL)
port.openPort()
port.setBaudRate(BAUDRATE)

def w1(mid, addr, val):
    ph.write1ByteTxRx(port, mid, addr, val)

def w4(mid, addr, val):
    v = int(val) & 0xFFFFFFFF
    ph.write4ByteTxRx(port, mid, addr, v)

def r4s(mid, addr):
    raw, _, _ = ph.read4ByteTxRx(port, mid, addr)
    return raw - 0x100000000 if raw > 0x7FFFFFFF else raw

# Setup: extended position mode, slow profile velocity, torque on
for mid in MOTOR_IDS:
    w1(mid, ADDR_TORQUE,   0)
    w1(mid, ADDR_OP_MODE,  4)   # extended position
    w4(mid, ADDR_PROF_VEL, 40)  # ≈ 9 rpm — slow and measurable
    w1(mid, ADDR_TORQUE,   1)

# Record home counts
home = {mid: r4s(mid, ADDR_PRES_POS) for mid in MOTOR_IDS}
print(f"\nCable per count : {1000/COUNTS_PER_M:.4f} mm")
print(f"Target delta    : {DELTA} counts  ({CABLE_MM} mm)\n")
print(f"{'Motor':<8} {'Home':>8} {'Goal':>8} {'Expected mm':>12}")

# Send goals: positive SIGNS wind in, negative pay out
# To DISPENSE cable, we unwind → invert the sign (paying out)
goals = {}
for mid in MOTOR_IDS:
    goals[mid] = home[mid] + (-SIGNS[mid] * DELTA)   # pay out = opposite of wind-in
    print(f"  M{mid}    {home[mid]:>8}  {goals[mid]:>8}  {CABLE_MM:>11.1f}")

# Sync Write all goals simultaneously
sw = GroupSyncWrite(port, ph, ADDR_GOAL_POS, 4)
for mid in MOTOR_IDS:
    v = int(goals[mid]) & 0xFFFFFFFF
    sw.addParam(mid, [v & 0xFF, (v>>8)&0xFF, (v>>16)&0xFF, (v>>24)&0xFF])
sw.txPacket()
sw.clearParam()

# Wait for motion then read back
import time
time.sleep(4)

print(f"\n{'Motor':<8} {'Goal':>8} {'Actual':>8} {'Error cts':>10} {'Error mm':>10}")
for mid in MOTOR_IDS:
    actual = r4s(mid, ADDR_PRES_POS)
    err_cts = actual - goals[mid]
    err_mm  = err_cts / COUNTS_PER_M * 1000
    print(f"  M{mid}    {goals[mid]:>8}  {actual:>8}  {err_cts:>+10}  {err_mm:>+9.2f} mm")

# Disable torque
for mid in MOTOR_IDS:
    w1(mid, ADDR_TORQUE, 0)

port.closePort()