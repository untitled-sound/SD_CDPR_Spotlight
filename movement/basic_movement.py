from dynamixel_sdk import *
import time
import threading
import os
import sys
import pygame

try:
    from dynamixel_sdk import (
        PortHandler,
        PacketHandler,
        GroupSyncWrite,
        GroupSyncRead,
        COMM_SUCCESS,
        DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD,
    )
except:
    sys.exit("dynamixel_sdk not found")

DEVICE       = "/dev/ttyUSB0"
BAUDRATE    = 1_000_000
PROTOCOL     = 2.0

MOTOR_IDS = [1, 2, 3, 4]

# XL330 Control Table addresses
ADDR_OP_MODE       = 11
ADDR_TORQUE_EN     = 64
ADDR_PROFILE_VEL   = 112
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

portHandler = PortHandler(DEVICE)
packetHandler = PacketHandler(PROTOCOL)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() > 0:
    controller = pygame.joystick.Joystick(0)
    controller.init()
    print(f"Controller '{controller.get_name()}' initialized.")
else:
    print("No controller found.")
    exit()

_state_lock  = threading.Lock()
_input_state = {"type": None, "value": 0.0, "axis": 0}


def _pack4(value: int) -> list:
    return [
        DXL_LOBYTE(DXL_LOWORD(value)),
        DXL_HIBYTE(DXL_LOWORD(value)),
        DXL_LOBYTE(DXL_HIWORD(value)),
        DXL_HIBYTE(DXL_HIWORD(value)),
    ]


def getInputs():
    # while True:
    #     for event in pygame.event.get():
    #         inputType = event.type
    #         if(event.type == pygame.JOYAXISMOTION):
    #             inputValue = event.value
    #             inputAxis = event.axis
    #     return inputType, inputValue, inputAxis
    while True:
        for event in pygame.event.get():
            with _state_lock:
                _input_state["type"] = event.type
                if event.type == pygame.JOYAXISMOTION:
                    _input_state["value"] = event.value
                    _input_state["axis"]  = event.axis
                elif event.type == pygame.JOYHATMOTION:
                    _input_state["value"] = event.value   # tuple e.g. (0, 1)
                    _input_state["axis"]  = -1
        time.sleep(0.01)   # 100 Hz poll — keeps CPU usage low
             

def setMode(mode): 
    for dxl_id in MOTOR_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_EN, TORQUE_OFF)
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OP_MODE, mode)
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_EN, TORQUE_ON)

        #packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROFILE_VEL, 50)

def setVelocity(velocity):
    sw = GroupSyncWrite(portHandler, packetHandler, ADD_GOAL_VELOCITY, 4)
    for dxl_id, rpm in zip(MOTOR_IDS, velocity):
        raw = int(rpm/0.229)
        sw.addParam(dxl_id, _pack4(raw & 0xFFFFFFFF))
    sw.txPacket()
    sw.clearParam()

setMode(MODE_VELOCITY)

# threadInput = threading.Thread(target=getInputs)
# threadMotors = threading.Thread(target=setVelocity)

threadInput = threading.Thread(target=getInputs, daemon=True)
threadInput.start()

while True: 
    with _state_lock:
        event_type = _input_state["type"]
        val        = _input_state["value"]
        axis       = _input_state["axis"]

    velInput = abs(val) * 60 if isinstance(val, float) else 0
 
    # For Reeling In
    # M1 (+) CW
    # M2 (-) CCW 
    # M3 (+) CW
    # M4 (-) CCW

    if event_type == pygame.JOYAXISMOTION:
        if val > 0.1:
            if   axis == 1: setVelocity([velInput,  velInput, -velInput,  velInput])
            elif axis == 2: setVelocity([ -velInput,  velInput, velInput, -velInput])
        elif val < -0.1:                              
            if   axis == 1: setVelocity([-velInput, -velInput,  velInput, velInput])
            elif axis == 2: setVelocity([velInput, -velInput,  -velInput,  velInput])
        else:
            setVelocity([0.0, 0.0, 0.0, 0.0])
 
    elif event_type == pygame.JOYHATMOTION:
        if   val == (0,  1): setVelocity([ 50,  -50,  50,  -50])
        elif val == (0, -1): setVelocity([-50, 50, -50, 50])
        else:                setVelocity([ 0.0, 0.0, 0.0, 0.0])
 
    else:
        setVelocity([0.0, 0.0, 0.0, 0.0])


setVelocity([-30.0, 30.0, -30.0, 30.0])
time.sleep(3)
setVelocity([0.0, 0.0])

