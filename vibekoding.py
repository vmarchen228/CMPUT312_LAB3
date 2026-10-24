#!/usr/bin/env python3
import math
import time
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, SpeedPercent

# Robot configuration (same as Lab 2)
upper_motor = LargeMotor(OUTPUT_A)
bottom_motor = LargeMotor(OUTPUT_B)

L1 = 11.5                     # cm
L2 = 9.25                     # cm
GEAR_RATIO = 40.0 / 24.0
MOTOR_SPEED = 9               # percent
LINE_STEP_CM = 0.5            # maximum spacing of waypoints
MOTOR_TIMEOUT_S = 10.0

# Adjust these to match the actual physical joint limits.
THETA1_MIN, THETA1_MAX = -180.0, 180.0
THETA2_MIN, THETA2_MAX = -180.0, 180.0

upper_motor.ramp_up_sp = 100
upper_motor.ramp_down_sp = 100
bottom_motor.ramp_up_sp = 100
bottom_motor.ramp_down_sp = 100


def zero_arm():
    input("Place the arm in its zero position, then press Enter...")
    upper_motor.position = 0
    bottom_motor.position = 0


def ik_candidates(x, y):
    """Return the two analytical IK solutions as (theta1, theta2) in degrees."""
    c2 = (x*x + y*y - L1*L1 - L2*L2) / (2.0 * L1 * L2)
    if c2 < -1.0000001 or c2 > 1.0000001:
        return []
    c2 = max(-1.0, min(1.0, c2))

    solutions = []
    for theta2 in (math.acos(c2), -math.acos(c2)):
        theta1 = math.atan2(y, x) - math.atan2(
            L2 * math.sin(theta2), L1 + L2 * math.cos(theta2)
        )
        a1 = (math.degrees(theta1) + 180.0) % 360.0 - 180.0
        a2 = (math.degrees(theta2) + 180.0) % 360.0 - 180.0
        if THETA1_MIN <= a1 <= THETA1_MAX and THETA2_MIN <= a2 <= THETA2_MAX:
            solutions.append((a1, a2))
    return solutions


def prepare_line_trajectory(x1, y1, x2, y2):
    """Interpolate the Cartesian line and calculate joint angles for each point."""
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 0.01:
        raise ValueError("Start and end points must be different")

    n = max(1, int(math.ceil(length / LINE_STEP_CM)))
    old_a1 = float(upper_motor.position)
    old_a2 = (float(bottom_motor.position) - old_a1) / GEAR_RATIO
    path = []
    branch = None

    for i in range(n + 1):
        t = float(i) / n
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        solutions = ik_candidates(x, y)
        if branch is not None:
            solutions = [a for a in solutions if a[1] * branch >= -0.0001]
        if not solutions:
            raise ValueError("Unreachable point: ({:.2f}, {:.2f}) cm".format(x, y))

        chosen = min(solutions, key=lambda a:
                     (a[0] - old_a1)**2 + (a[1] - old_a2)**2)
        if branch is None:
            branch = 1 if chosen[1] >= 0 else -1
        if path and max(abs(chosen[0] - old_a1), abs(chosen[1] - old_a2)) > 25.0:
            raise ValueError("Sudden joint-angle jump; choose another path")

        path.append(chosen)
        old_a1, old_a2 = chosen
    return path


def move_both_motors(theta1, theta2):
    """Start both motors together, using the Lab 2 gear mapping."""
    upper_target = theta1
    bottom_target = theta1 + GEAR_RATIO * theta2
    du = abs(upper_target - upper_motor.position)
    db = abs(bottom_target - bottom_motor.position)
    distance = max(du, db)
    if distance < 0.5:
        return

    if du >= 0.5:
        upper_motor.on_to_position(
            SpeedPercent(max(3.0, MOTOR_SPEED * du / distance)),
            upper_target, brake=True, block=False
        )
    if db >= 0.5:
        bottom_motor.on_to_position(
            SpeedPercent(max(3.0, MOTOR_SPEED * db / distance)),
            bottom_target, brake=True, block=False
        )

    deadline = time.time() + MOTOR_TIMEOUT_S
    while time.time() < deadline:
        upper_done = du < 0.5 or abs(upper_motor.position - upper_target) <= 1.5
        bottom_done = db < 0.5 or abs(bottom_motor.position - bottom_target) <= 1.5
        if upper_done and bottom_done:
            return
        time.sleep(0.03)

    upper_motor.off(brake=True)
    bottom_motor.off(brake=True)
    raise RuntimeError("Motor timeout")


def draw_straight_line(x1, y1, x2, y2):
    try:
        path = prepare_line_trajectory(x1, y1, x2, y2)
    except ValueError as exc:
        print("Cannot draw line:", exc)
        return

    print("Line length: {:.2f} cm; segments: {}".format(
        math.hypot(x2 - x1, y2 - y1), len(path) - 1
    ))
    input("Lift the pen OFF the paper, then press Enter...")
    try:
        move_both_motors(*path[0])
        input("Lower the pen ONTO the paper, then press Enter...")
        for angles in path[1:]:
            move_both_motors(*angles)
        print("Straight line complete. Lift the pen.")
    except (RuntimeError, KeyboardInterrupt) as exc:
        upper_motor.off(brake=True)
        bottom_motor.off(brake=True)
        print("Movement stopped:", exc)


def main():
    zero_arm()
    while True:
        try:
            start = input("Start point x1 y1 (or q): ").strip()
            if start.lower() == "q":
                break
            x1, y1 = map(float, start.split())
            x2, y2 = map(float, input("End point x2 y2: ").split())
            draw_straight_line(x1, y1, x2, y2)
        except ValueError:
            print("Enter two coordinates, e.g. 14 0")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
