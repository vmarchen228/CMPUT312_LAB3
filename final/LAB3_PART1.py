#!/usr/bin/env python3
# RUN ON THE BRICK

import math
import time
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, SpeedDPS


# ---------------- settings ----------------

upper_motor = LargeMotor(OUTPUT_A)    # drives theta1 directly
bottom_motor = LargeMotor(OUTPUT_B)   # drives theta2 through the 40/24 gears (coupled to theta1)

L1 = 11.5               # length of link 1 (cm)
L2 = 9.25               # length of link 2 (cm)
GEAR = 40.0 / 24.0      # bottom motor degrees per degree of theta2

STEP_CM = 0.25          # distance between waypoints along the line (cm)
MAX_DPS = 100           # max motor speed
MIN_DPS = 5             # minimum motor speed to prevent stalling
SEGMENT_TIMEOUT = 5.0   # stop trying to move after this many seconds

# joint limits in degrees (to be replaced with limits from calibration)
THETA1_MIN = -180.0
THETA1_MAX = 180.0
THETA2_MIN = -180.0
THETA2_MAX = 180.0

# no acceleration ramps
upper_motor.ramp_up_sp = 0
upper_motor.ramp_down_sp = 0
bottom_motor.ramp_up_sp = 0
bottom_motor.ramp_down_sp = 0


# ---------------- kinematics ----------------

def forward_kinematics(theta1_deg, theta2_deg):
    """Joint angles (degrees) -> pen position (x, y) in cm."""

    theta1 = math.radians(theta1_deg)
    theta2 = math.radians(theta2_deg)

    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)

    return x, y


def inverse_kinematics(x, y, elbow):
    """Pen position (x, y) in cm -> joint angles (degrees), or None if unreachable.
    elbow = +1 or -1 picks which of the two solutions (elbow bent one way or the other)."""

    # law of cosines gives cos(theta2) from the distance to the target
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2.0 * L1 * L2)

    # outside [-1, 1] means no triangle exists: the point is out of reach
    if cos_theta2 < -1.0 or cos_theta2 > 1.0:
        return None

    theta2 = elbow * math.acos(cos_theta2)

    # angle to the target minus the extra angle caused by the bent elbow
    angle_to_target = math.atan2(y, x)
    elbow_offset = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = angle_to_target - elbow_offset

    return math.degrees(theta1), math.degrees(theta2)


def read_joints():
    """Current joint angles (degrees) from the motor encoders."""

    theta1 = upper_motor.position

    # the bottom motor also turns when theta1 moves, so subtract theta1 before converting through the gears
    theta2 = (bottom_motor.position - theta1) / GEAR

    return theta1, theta2


def in_limits(theta1, theta2):
    """True if both angles are inside the joint limits."""

    theta1_ok = THETA1_MIN <= theta1 <= THETA1_MAX
    theta2_ok = THETA2_MIN <= theta2 <= THETA2_MAX

    return theta1_ok and theta2_ok


# ---------------- path planning ----------------

def plan_line(x1, y1, x2, y2, elbow, start_theta1):
    """Split the line into waypoints STEP_CM apart and solve IK for each one.
    Returns (path, None) where path is a list of (theta1, theta2), or (None, reason) if a point can't be reached."""

    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx**2 + dy**2)

    # number of segments; at least 1 so a very short line still works
    segment_count = int(math.ceil(length / STEP_CM))
    if segment_count < 1:
        segment_count = 1

    path = []
    previous_theta1 = start_theta1

    for i in range(segment_count + 1):

        # fraction of the way along the line: 0 at the start, 1 at the end
        fraction = i / segment_count
        x = x1 + fraction * dx
        y = y1 + fraction * dy

        solution = inverse_kinematics(x, y, elbow)
        if solution is None:
            reason = "({:.2f}, {:.2f}) is outside the workspace".format(x, y)
            return None, reason

        theta1 = solution[0]
        theta2 = solution[1]

        # atan2 jumps by 360 degrees at +-180; shift theta1 so it stays close to the previous waypoint
        while theta1 - previous_theta1 > 180.0:
            theta1 = theta1 - 360.0
        while theta1 - previous_theta1 < -180.0:
            theta1 = theta1 + 360.0

        if not in_limits(theta1, theta2):
            reason = "({:.2f}, {:.2f}) requires theta1 = {:.1f}, and theta2 = {:.1f}, which is outside joint limits".format(x, y, theta1, theta2)
            return None, reason

        path.append((theta1, theta2))
        previous_theta1 = theta1

    return path, None


def choose_path(x1, y1, x2, y2):
    """Try both elbow directions and choose the path closest to the current position.
    Returns the path, elbow direction, and any errors."""

    current_theta1, current_theta2 = read_joints()

    best_path = None
    best_elbow = None
    best_cost = None
    reasons = []

    for elbow in (+1, -1):

        path, reason = plan_line(x1, y1, x2, y2, elbow, current_theta1)

        if path is None:
            reasons.append("elbow {:+d}: {}".format(elbow, reason))
            continue

        # Calculate how much both motors need to move to reach the start
        first_theta1 = path[0][0]
        first_theta2 = path[0][1]
        cost = abs(first_theta1 - current_theta1) + abs(first_theta2 - current_theta2)

        if best_path is None or cost < best_cost:
            best_path = path
            best_elbow = elbow
            best_cost = cost

    return best_path, best_elbow, reasons


# ---------------- motion ----------------

def move_to_joints(theta1, theta2):
    """Move both motors at the same time, with speeds scaled so they finish together."""

    upper_target = theta1
    bottom_target = theta1 + GEAR * theta2      # bottom motor must also follow theta1 because of the gear coupling

    upper_distance = abs(upper_target - upper_motor.position)
    bottom_distance = abs(bottom_target - bottom_motor.position)

    # The motor with the longer move runs at MAX_DPS, the other one moves proportionally slower,
    # so both arrive at the same time and the pen moves in a straight segment
    longest = max(upper_distance, bottom_distance)
    if longest < 0.000001:
        longest = 0.000001

    upper_speed = MAX_DPS * upper_distance / longest
    bottom_speed = MAX_DPS * bottom_distance / longest

    if upper_speed < MIN_DPS:
        upper_speed = MIN_DPS
    if bottom_speed < MIN_DPS:
        bottom_speed = MIN_DPS

    # brake=True holds the position after the move
    upper_motor.on_to_position(SpeedDPS(upper_speed), upper_target, brake=True, block=False)
    bottom_motor.on_to_position(SpeedDPS(bottom_speed), bottom_target, brake=True, block=False)

    time.sleep(0.02)    # short delay gives the motors time to start

    start_time = time.time()
    while 'running' in upper_motor.state or 'running' in bottom_motor.state:
        if time.time() - start_time > SEGMENT_TIMEOUT:
            print("Segment timed out (stalled?)")
            break
        time.sleep(0.01)


def draw_line(x1, y1, x2, y2):
    """Plan the line, move to its start, wait for the pen, then draw it."""

    path, elbow, reasons = choose_path(x1, y1, x2, y2)

    if path is None:
        print("Can't draw this line:")
        for reason in reasons:
            print("  " + reason)
        return

    print("Planned {} waypoints ({:.2f} cm apart), elbow {:+d}".format(len(path), STEP_CM, elbow))

    # Go to the start 
    start_theta1 = path[0][0]
    start_theta2 = path[0][1]
    move_to_joints(start_theta1, start_theta2)
    input("At start point. Put the pen down and press Enter...")

    start_time = time.time()

    for waypoint in path[1:]:
        theta1 = waypoint[0]
        theta2 = waypoint[1]
        move_to_joints(theta1, theta2)

    duration = time.time() - start_time

    # End position estimated from the encoders to compare with the target
    end_theta1, end_theta2 = read_joints()
    end_x, end_y = forward_kinematics(end_theta1, end_theta2)
    error = math.sqrt((end_x - x2)**2 + (end_y - y2)**2)

    print("Done in {:.1f} s".format(duration))
    print("Target end:  ({:.2f}, {:.2f}) cm".format(x2, y2))
    print("Encoder end: ({:.2f}, {:.2f}) cm  -> error {:.2f} cm".format(end_x, end_y, error))


# ---------------- main ----------------

def main():
    input("Place the arm in the zero position (theta1 = 0, theta2 = 0) and press Enter...")
    upper_motor.position = 0
    bottom_motor.position = 0

    try:
        while True:
            text = input("\nEnter x1 y1 x2 y2 in cm (or q to quit): ").strip()

            if text.lower() == 'q':
                break

            parts = text.split()
            if len(parts) != 4:
                print("Need exactly four numbers.")
                continue

            try:
                x1 = float(parts[0])
                y1 = float(parts[1])
                x2 = float(parts[2])
                y2 = float(parts[3])
            except ValueError:
                print("Need exactly four numbers.")
                continue

            draw_line(x1, y1, x2, y2)

    finally:
        upper_motor.off(brake=False)
        bottom_motor.off(brake=False)


if __name__ == '__main__':
    main()