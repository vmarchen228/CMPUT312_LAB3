#!/usr/bin/python3
# RUN ON BRICK

import socket
import os
import time
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, SpeedPercent

upper_motor = LargeMotor(OUTPUT_A)
bottom_motor = LargeMotor(OUTPUT_B)

MOTOR_SPEED = 20
LIMIT_MARGIN = 5.0

upper_motor.ramp_up_sp = 100
upper_motor.ramp_down_sp = 100
bottom_motor.ramp_up_sp = 100
bottom_motor.ramp_down_sp = 100

def current_angles():
    theta1 = upper_motor.position
    theta2 = (bottom_motor.position - theta1) * (24.0 / 40.0)
    return theta1, theta2

def calibrate_limits():
    global THETA1_MIN, THETA1_MAX, THETA2_MIN, THETA2_MAX

    print("\n================ CALIBRATE JOINT LIMITS ================\n")
    upper_motor.off(brake=False)
    bottom_motor.off(brake=False)

    try:
        input("Move theta1 to the first SAFE limit and press Enter...")
        theta1_a, _ = current_angles()

        input("Move theta1 to the other SAFE limit and press Enter...")
        theta1_b, _ = current_angles()

        input("Move theta2 to the first SAFE limit and press Enter...")
        _, theta2_a = current_angles()

        input("Move theta2 to the other SAFE limit and press Enter...")
        _, theta2_b = current_angles()

        theta1_min = min(theta1_a, theta1_b) + LIMIT_MARGIN
        theta1_max = max(theta1_a, theta1_b) - LIMIT_MARGIN
        theta2_min = min(theta2_a, theta2_b) + LIMIT_MARGIN
        theta2_max = max(theta2_a, theta2_b) - LIMIT_MARGIN

        if theta1_min >= theta1_max or theta2_min >= theta2_max:
            raise RuntimeError("Invalid calibrated joint limits")

        THETA1_MIN, THETA1_MAX = theta1_min, theta1_max
        THETA2_MIN, THETA2_MAX = theta2_min, theta2_max

        print("Theta1: {:.2f} to {:.2f} deg".format(THETA1_MIN, THETA1_MAX))
        print("Theta2: {:.2f} to {:.2f} deg".format(THETA2_MIN, THETA2_MAX))

        input("Move BOTH joints inside these safe ranges and press Enter...")
        theta1, theta2 = current_angles()
        if not (THETA1_MIN <= theta1 <= THETA1_MAX and
                THETA2_MIN <= theta2 <= THETA2_MAX):
            raise RuntimeError("Starting position is outside calibrated limits")

    finally:
        upper_motor.off(brake=True)
        bottom_motor.off(brake=True)

def move_to_angles(theta1, theta2):
    if not THETA1_MIN <= theta1 <= THETA1_MAX:
        raise ValueError("Theta1 {:.2f} outside limits [{:.2f}, {:.2f}]".format(
            theta1, THETA1_MIN, THETA1_MAX))
    if not THETA2_MIN <= theta2 <= THETA2_MAX:
        raise ValueError("Theta2 {:.2f} outside limits [{:.2f}, {:.2f}]".format(
            theta2, THETA2_MIN, THETA2_MAX))

    bottom_angle = theta1 + (40.0 / 24.0) * theta2

    upper_motor.on_to_position(
        SpeedPercent(MOTOR_SPEED),
        theta1,
        brake=False,
        block=False
    )

    bottom_motor.on_to_position(
        SpeedPercent(MOTOR_SPEED),
        bottom_angle,
        brake=False,
        block=False
    )

    upper_ok = upper_motor.wait_until_not_moving(timeout=10000)
    bottom_ok = bottom_motor.wait_until_not_moving(timeout=10000)

    if (not upper_ok or not bottom_ok
            or upper_motor.is_stalled or bottom_motor.is_stalled
            or abs(upper_motor.position - theta1) > 3
            or abs(bottom_motor.position - bottom_angle) > 3):
        raise RuntimeError("Motor failed to reach target")

    print("Movement completed")

# This class handles the client side of communication. It has a set of predefined messages to send to the server as well as functionality to poll and decode data.
class Client:
    def __init__(self, host, port):
        # We need to use the ipv4 address that shows up in ipconfig in the computer for the USB. Ethernet adapter handling the connection to the EV3
        print("Setting up client\nAddress: " + host + "\nPort: " + str(port))
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))

    # Block until a message from the server is received. When the message is received it will be decoded and returned as a string.
    # Output: UTF-8 decoded string containing the instructions from server.
    def pollData(self):
        print("Waiting for Data")
        data = self.s.recv(128).decode("UTF-8")
        print("Data Received")
        return data

    # Sends a message to the server letting it know that the movement of the motors was executed without any inconvenience.
    def sendDone(self):
        self.s.send("DONE".encode("UTF-8"))

    # Sends a message to the server letting it know that there was an isse during the execution of the movement (obstacle avoided) and that the initial jacobian should be recomputed (Visual servoing started from scratch)
    def sendReset(self):
        self.s.send("RESET".encode("UTF-8"))


host = "192.168.0.2"
port = 9999

upper_motor.off(brake=False)
bottom_motor.off(brake=False)
input("Place arm in zero position and press Enter...")
upper_motor.position = 0
bottom_motor.position = 0
calibrate_limits()

client = Client(host, port)
i = 0
while True:
    # print(client.pollData())
    # time.sleep(1)
    # client.sendDone()
    data = client.pollData()
    if not data or data == "EXIT":
        break

    try:
        theta1_delta, theta2_delta = map(float, data.split(","))
        theta1_current, theta2_current = current_angles()
        move_to_angles(theta1_current + theta1_delta, theta2_current + theta2_delta)
        client.sendDone()

    except (ValueError, RuntimeError) as e:
        upper_motor.off(brake=False)
        bottom_motor.off(brake=False)
        print("Error:", e)
        client.sendReset()
        continue
