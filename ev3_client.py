#!/usr/bin/env python3
"""Run on EV3 (ev3dev2). Receives absolute joint angles in degrees."""
import socket
import time
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, SpeedPercent

SERVER_IP = '192.168.0.2'  # Change to your laptop's USB/network IP
PORT = 9999
GEAR_RATIO = 40.0 / 24.0
MOTOR_SPEED = 9  # percent
TIMEOUT = 10.0

upper_motor = LargeMotor(OUTPUT_A)
bottom_motor = LargeMotor(OUTPUT_B)


def stop_motors():
    upper_motor.off(brake=True)
    bottom_motor.off(brake=True)


def move_to_joint_angles(theta1, theta2):
    """Lab 2 mapping: motor A = theta1; motor B = theta1 + ratio * theta2."""
    target_a = theta1
    target_b = theta1 + GEAR_RATIO * theta2
    distance_a = abs(target_a - upper_motor.position)
    distance_b = abs(target_b - bottom_motor.position)
    maximum = max(distance_a, distance_b)
    if maximum < 0.5:
        return

    if distance_a >= 0.5:
        upper_motor.on_to_position(
            SpeedPercent(max(3.0, MOTOR_SPEED * distance_a / maximum)),
            target_a, brake=True, block=False)
    if distance_b >= 0.5:
        bottom_motor.on_to_position(
            SpeedPercent(max(3.0, MOTOR_SPEED * distance_b / maximum)),
            target_b, brake=True, block=False)

    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        a_done = distance_a < 0.5 or abs(upper_motor.position - target_a) <= 1.5
        b_done = distance_b < 0.5 or abs(bottom_motor.position - target_b) <= 1.5
        if a_done and b_done:
            stop_motors()
            return
        time.sleep(0.03)
    stop_motors()
    raise RuntimeError('Motor movement timed out')


def main():
    input('Put the arm in its Lab 2 zero position, then press Enter...')
    upper_motor.position = 0
    bottom_motor.position = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print('Connecting to {}:{}...'.format(SERVER_IP, PORT))
        sock.connect((SERVER_IP, PORT))
        with sock.makefile('r', encoding='utf-8') as commands:
            for line in commands:
                command = line.strip()
                if command == 'EXIT':
                    break
                try:
                    theta1, theta2 = map(float, command.split(','))
                    print('Moving to joints: {}, {}'.format(theta1, theta2))
                    move_to_joint_angles(theta1, theta2)
                    sock.sendall(b'DONE\n')
                except (ValueError, RuntimeError) as exc:
                    stop_motors()
                    print('Movement error:', exc)
                    sock.sendall(b'ERROR\n')
    stop_motors()
    print('Disconnected')


if __name__ == '__main__':
    try:
        main()
    finally:
        stop_motors()
