import numpy as np
import time
from dynamixel_sdk import *

PORT = "/dev/ttyUSB0"
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

MOTOR_IDS = [1, 2, 3, 4]

# XL330 Control Table addresses
ADDR_OP_MODE       = 11
ADDR_TORQUE_EN     = 64
ADDR_PROF_VEL   = 112
ADD_GOAL_VELOCITY = 104
ADDR_GOAL_POS      = 116   # 4-byte signed (extended mode)
ADDR_PRESENT_POS   = 132   # 4-byte signed (extended mode)
ADDR_PRESENT_CUR   = 146   # 2-byte signed [mA]

MODE_CURRENT = 0
MODE_VELOCITY = 1
MODE_POSITION = 3
MODE_EXTENDED_POS  = 4

TORQUE_ON          = 1
TORQUE_OFF         = 0

# Conversion
COUNTS_PER_M = 13038 

# +1 = CW tightens, -1 = CCW tightens
SIGNS = {1: +1, 2: -1, 3: +1, 4: -1}

anchors = {
    1: np.array([0.00, 0.00, 2.00]),
    2: np.array([0.00, 1.48, 2.00]),
    3: np.array([1.43, 1.48, 2.00]),
    4: np.array([1.43, 0.00, 2.00]),
}

# center (ee)
center = np.array([1.43/2, 1.48/2, 1.00])

# target
target = np.array([1.2, 1.48/2, 1.00])

# IK 
def compute_lengths(p):
    return {i: np.linalg.norm(p - anchors[i]) for i in MOTOR_IDS}

portHandler = PortHandler(PORT)
packetHandler = PacketHandler(PROTOCOL_VERSION)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

# Set position mode + enable torque
for dxl_id in MOTOR_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_EN, TORQUE_OFF)
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OP_MODE, MODE_POSITION)
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROF_VEL, 40 & 0xFFFFFFFF)  # ≈ 9 rpm — slow 
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_EN, TORQUE_ON)

start_pos = {}

for dxl_id in MOTOR_IDS:
    pos, _, _ = packetHandler.read4ByteTxRx(
        portHandler, dxl_id, ADDR_PRESENT_POS
    )
    start_pos[dxl_id] = pos

print("\nStart Positions (counts):")
for i in MOTOR_IDS:
    print(f"Motor {i}: {start_pos[i]}")

L_center = compute_lengths(center)

def move_to(p):
    L_target = compute_lengths(p)

    for dxl_id in MOTOR_IDS:
        delta_L = L_target[dxl_id] - L_center[dxl_id]
        delta_counts = int(SIGNS[dxl_id] * delta_L * COUNTS_PER_M)

        goal_position = start_pos[dxl_id] + delta_counts

        print(f"Motor {dxl_id}:")
        print(f"  ΔL = {delta_L:.4f} m")
        print(f"  Δcounts = {delta_counts}")
        print(f"  Goal = {goal_position}")

        packetHandler.write4ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_GOAL_POS,
            goal_position
        )

print("Moving to target")
move_to(target)
time.sleep(7)

print("\nReturning to center")
move_to(center)
time.sleep(7)

for dxl_id in MOTOR_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_EN, TORQUE_OFF)
