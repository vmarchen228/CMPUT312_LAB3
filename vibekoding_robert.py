#!/usr/bin/env python3
# Lab 3, Part 1.1: draw a straight line between two points.
# RUN ON THE BRICK. No PC, server or camera needed for this part.
# Uses the same arm geometry and gearing as lab2_main.py.

import math
import time
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, SpeedDPS

upper_motor = LargeMotor(OUTPUT_A)    # theta1
bottom_motor = LargeMotor(OUTPUT_B)   # theta2, through the 40/24 gears (coupled to theta1)

L1 = 11.5
L2 = 9.25
GEAR = 40.0 / 24.0

STEP_CM = 0.25      # spacing between waypoints along the line (report Q1.1: step size)
MAX_DPS = 100      # speed of the faster motor on each small segment, deg/s
MIN_DPS = 5        # slower motor never goes below this (very low speeds stall)
SEGMENT_TIMEOUT = 5.0

# Paste the limits you measured with "c" in lab2_main.py (motor degrees of each joint).
THETA1_MIN, THETA1_MAX = -180.0, 180.0
THETA2_MIN, THETA2_MAX = -180.0, 180.0

LOG_FILE = "line_trials.csv"

for m in (upper_motor, bottom_motor):
    m.ramp_up_sp = 0      # lab 2 used 1500 ms ramps; with many tiny moves that makes it crawl and jerk
    m.ramp_down_sp = 0


# ---------------- kinematics ----------------

def forward_kinematics(t1_deg, t2_deg):
    t1 = math.radians(t1_deg)
    t2 = math.radians(t2_deg)
    x = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
    y = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
    return x, y


def inverse_kinematics(x, y, elbow):
    """elbow = +1 or -1 picks which of the two IK solutions (elbow one way or the other)."""
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if c2 < -1.0 or c2 > 1.0:
        return None
    t2 = elbow * math.acos(c2)
    t1 = math.atan2(y, x) - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))
    return math.degrees(t1), math.degrees(t2)


def read_joints():
    a = upper_motor.position
    b = bottom_motor.position
    return a, (b - a) / GEAR          # same conversion as current_position() in lab 2


def in_limits(t1, t2):
    return THETA1_MIN <= t1 <= THETA1_MAX and THETA2_MIN <= t2 <= THETA2_MAX


# ---------------- planning ----------------

def plan_line(x1, y1, x2, y2, elbow, start_t1):
    """Split the line into waypoints STEP_CM apart and solve IK for each.
    Returns a list of (x, y, theta1, theta2), or (None, reason) if any point is unreachable."""
    length = math.hypot(x2 - x1, y2 - y1)
    n = max(1, int(math.ceil(length / STEP_CM)))
    path = []
    prev_t1 = start_t1
    for i in range(n + 1):
        s = i / n
        x = x1 + s * (x2 - x1)
        y = y1 + s * (y2 - y1)
        sol = inverse_kinematics(x, y, elbow)
        if sol is None:
            return None, "({:.2f}, {:.2f}) is outside the workspace".format(x, y)
        t1, t2 = sol
        # atan2 jumps by 360 deg at +-180; keep theta1 continuous along the path
        while t1 - prev_t1 > 180.0:
            t1 -= 360.0
        while t1 - prev_t1 < -180.0:
            t1 += 360.0
        if not in_limits(t1, t2):
            return None, "({:.2f}, {:.2f}) needs theta1={:.1f}, theta2={:.1f}, outside joint limits".format(x, y, t1, t2)
        path.append((x, y, t1, t2))
        prev_t1 = t1
    return path, None


def choose_path(x1, y1, x2, y2):
    """Try both elbow configurations, keep the valid one closest to where the arm is now."""
    cur_t1, cur_t2 = read_joints()
    best, reasons = None, []
    for elbow in (+1, -1):
        path, reason = plan_line(x1, y1, x2, y2, elbow, cur_t1)
        if path is None:
            reasons.append("elbow {:+d}: {}".format(elbow, reason))
            continue
        cost = abs(path[0][2] - cur_t1) + abs(path[0][3] - cur_t2)
        if best is None or cost < best[0]:
            best = (cost, path, elbow)
    if best is None:
        return None, None, reasons
    return best[1], best[2], reasons


# ---------------- motion ----------------

def move_to_joints(t1, t2):
    """Move both motors at the same time, with speeds scaled so they finish together."""
    a_target = t1
    b_target = t1 + GEAR * t2          # same coupling as move_to_angles() in lab 2
    da = abs(a_target - upper_motor.position)
    db = abs(b_target - bottom_motor.position)
    big = max(da, db, 1e-6)
    sa = max(MIN_DPS, MAX_DPS * da / big)
    sb = max(MIN_DPS, MAX_DPS * db / big)

    # brake=True -> 'hold': the motors resist the pen dragging them between waypoints
    upper_motor.on_to_position(SpeedDPS(sa), a_target, brake=True, block=False)
    bottom_motor.on_to_position(SpeedDPS(sb), b_target, brake=True, block=False)

    time.sleep(0.02)
    t0 = time.time()
    while ('running' in upper_motor.state or 'running' in bottom_motor.state):
        if time.time() - t0 > SEGMENT_TIMEOUT:
            print("Segment timed out (stalled?)")
            break
        time.sleep(0.01)


def draw_line(x1, y1, x2, y2):
    path, elbow, reasons = choose_path(x1, y1, x2, y2)
    if path is None:
        print("Can't draw this line:")
        for r in reasons:
            print("  " + r)
        return

    print("Planned {} waypoints ({:.1f} cm apart), elbow {:+d}".format(len(path), STEP_CM, elbow))

    # go to the start with the pen lifted, otherwise it draws a curve on the way there
    _, _, t1, t2 = path[0]
    move_to_joints(t1, t2)
    input("At start point. Put the pen down and press Enter...")

    t_start = time.time()
    for (_, _, t1, t2) in path[1:]:
        move_to_joints(t1, t2)
    duration = time.time() - t_start

    # encoder-estimated end position, for the accuracy/repeatability table
    a1, a2 = read_joints()
    xe, ye = forward_kinematics(a1, a2)
    err = math.hypot(xe - x2, ye - y2)
    print("Done in {:.1f} s".format(duration))
    print("Target end:  ({:.2f}, {:.2f}) cm".format(x2, y2))
    print("Encoder end: ({:.2f}, {:.2f}) cm  -> error {:.2f} cm".format(xe, ye, err))
    print("Measure the drawn line with a ruler too; that's the real accuracy.")

    try:
        with open(LOG_FILE, "a") as f:
            f.write("{},{},{},{},{:.3f},{:.3f},{:.3f},{:.2f},{}\n".format(
                x1, y1, x2, y2, xe, ye, err, duration, STEP_CM))
    except OSError:
        pass


def main():
    input("Place the arm in the zero position (theta1 = 0, theta2 = 0) and press Enter...")
    upper_motor.position = 0
    bottom_motor.position = 0

    try:
        while True:
            s = input("\nEnter x1 y1 x2 y2 in cm (or q to quit): ").strip()
            if s.lower() == 'q':
                break
            try:
                x1, y1, x2, y2 = map(float, s.split())
            except ValueError:
                print("Need exactly four numbers.")
                continue
            draw_line(x1, y1, x2, y2)
    finally:
        upper_motor.off(brake=False)
        bottom_motor.off(brake=False)


if __name__ == '__main__':
    main()