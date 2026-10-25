#!/usr/bin/env python3
# LAB 3 - PART 2: UNCALIBRATED VISUAL SERVOING
# RUN ON THE LAPTOP. Start this first, then client.py on the brick.
#
# The brick moves each joint BY the angles we send (degrees) and replies "DONE"
# ("RESET" or "ERROR ..." means a motor stopped a few degrees off; the camera still sees where it went).
# Press ESC in the camera window to stop a run.

import socket
import time
import math
import cv2

HOST = "192.168.0.2"
PORT = 9999

# Colour ranges in HSV. Tune them with color_tracking_hsv.py.
END_EFFECTOR_LOW = (100, 150, 0)      # blue sticker on the end effector
END_EFFECTOR_HIGH = (140, 255, 255)
TARGET_LOW = (0, 125, 50)             # red target dot
TARGET_HIGH = (10, 255, 255)
MIN_MARKER_AREA = 30                  # pixels; smaller blobs are noise

FLUSH_FRAMES = 5        # frames skipped before measuring, so we never use a frame from during a move
SAMPLES = 5             # frames averaged per measurement
ESC_KEY = 27

PROBE_ANGLE = 10.0      # 2.1: degrees each joint moves to estimate the initial Jacobian
TOLERANCE = 10.0         # 2.2: stop when the end effector is this close to the target (pixels)
#TOLERANCE = 5.0         # 2.2: stop when the end effector is this close to the target (pixels)
MAX_ITERATIONS = 50     # 2.2: give up after this many moves
ALPHA = 0.5             # 2.3: Broyden update strength
MIN_PIXEL_CHANGE = 2.0  # 2.3: only update the Jacobian if the end effector moved more than this (pixels)
GAIN = 0.5              # 2.4: take half of the computed step each time
MAX_STEP = 15.0         # 2.4: no joint moves more than this per step (degrees)
OCCLUSION_RADIUS = 40.0 # 2.6: ignore target detections this close to the arm (the arm may be covering the dot)

camera = None
brick = None
stop_requested = False


# ---------------- camera ----------------

def find_marker(hsv, low, high):

    # White = pixels in the colour range
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.erode(mask, None, iterations=1)    # remove single-pixel noise
    mask = cv2.dilate(mask, None, iterations=2)   # fill small holes

    # [-2] works in both OpenCV 3 (returns 3 values) and OpenCV 4 (returns 2 values)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) == 0:
        return None

    biggest = max(contours, key=cv2.contourArea)

    # m00 = area, m10 = sum of x, m01 = sum of y  ->  centre = (m10 / m00, m01 / m00)
    moments = cv2.moments(biggest)
    if moments["m00"] < MIN_MARKER_AREA:
        return None

    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]

    return u, v


def draw_marker(frame, point, color):
    if point is not None:
        cv2.circle(frame, (int(point[0]), int(point[1])), 10, color, 2)


def read_frame():
    global stop_requested

    ok, frame = camera.read()
    if not ok:
        return None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    end_effector = find_marker(hsv, END_EFFECTOR_LOW, END_EFFECTOR_HIGH)
    target = find_marker(hsv, TARGET_LOW, TARGET_HIGH)

    draw_marker(frame, end_effector, (255, 0, 0))   # blue circle
    draw_marker(frame, target, (0, 0, 255))         # red circle
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ESC_KEY:
        stop_requested = True

    return end_effector, target


def average(points):
    if len(points) == 0:
        return None

    u = sum(point[0] for point in points) / len(points)
    v = sum(point[1] for point in points) / len(points)

    return u, v


def measure_markers():

    for i in range(FLUSH_FRAMES):
        read_frame()

    end_effector_points = []
    target_points = []

    for i in range(SAMPLES):
        end_effector, target = read_frame()

        if end_effector is not None:
            end_effector_points.append(end_effector)

        if target is not None:
            target_points.append(target)

    return average(end_effector_points), average(target_points)


def read_end_effector():
    for attempt in range(20):
        end_effector, target = measure_markers()
        if end_effector is not None:
            return end_effector

    print("End effector not found. Check the HSV range and lighting.")
    return None


def read_target():
    for attempt in range(20):
        end_effector, target = measure_markers()
        if target is not None:
            return target

    print("Target not found. Check the HSV range and lighting.")
    return None


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


# ---------------- brick ----------------

def connect_to_brick():
    print("\n================ WAITING FOR THE BRICK ================\n")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, PORT))
    listener.listen(1)

    print("Address: {}  Port: {}".format(HOST, PORT))
    print("Now run client.py on the brick...")

    connection, address = listener.accept()
    print("Connected to: " + str(address))

    return connection


def move_joints(theta1_delta, theta2_delta):
    print("Moving joints by theta1 = {:.2f} deg, theta2 = {:.2f} deg".format(theta1_delta, theta2_delta))

    brick.send("{},{}".format(theta1_delta, theta2_delta).encode("UTF-8"))

    # Wait for the brick's reply, but keep reading the camera meanwhile
    # so the video doesn't freeze while the arm is moving.
    brick.settimeout(0.001)
    while True:
        try:
            reply = brick.recv(128).decode("UTF-8")
            break
        except socket.timeout:
            read_frame()
    brick.settimeout(None)

    if reply == "":
        raise SystemExit("The brick disconnected.")

    if reply != "DONE":
        print("Warning from the brick: " + reply)


# ---------------- 2.1: initial Jacobian ----------------

def probe_joint(theta1_delta, theta2_delta):

    before = read_end_effector()
    move_joints(theta1_delta, theta2_delta)
    after = read_end_effector()

    # Move back, so both probes start from the same pose
    move_joints(-theta1_delta, -theta2_delta)

    if before is None or after is None:
        return None

    du = after[0] - before[0]
    dv = after[1] - before[1]

    return du, dv


def estimate_initial_jacobian():
    print("\n================ ESTIMATE INITIAL JACOBIAN ================\n")

    # Move theta1 only -> first column of J
    result1 = probe_joint(PROBE_ANGLE, 0.0)

    # Move theta2 only -> second column of J
    result2 = probe_joint(0.0, PROBE_ANGLE)

    if result1 is None or result2 is None:
        return None

    du1, dv1 = result1
    du2, dv2 = result2

    if math.sqrt(du1**2 + dv1**2) < 3 or math.sqrt(du2**2 + dv2**2) < 3:
        print("A probe barely moved the end effector. Move the arm away from its joint limits and try again.")
        return None

    # J = [ du/dtheta1   du/dtheta2 ]
    #     [ dv/dtheta1   dv/dtheta2 ]     (pixels per degree)
    j11 = du1 / PROBE_ANGLE
    j12 = du2 / PROBE_ANGLE
    j21 = dv1 / PROBE_ANGLE
    j22 = dv2 / PROBE_ANGLE

    J = [[j11, j12],
         [j21, j22]]

    print("Initial Jacobian (pixels per degree):")
    print_jacobian(J)

    return J


def print_jacobian(J):
    print("  [ {:8.3f}  {:8.3f} ]".format(J[0][0], J[0][1]))
    print("  [ {:8.3f}  {:8.3f} ]".format(J[1][0], J[1][1]))


# ---------------- 2.3: Broyden update ----------------

def broyden_update(J, du, dv, dtheta1, dtheta2):

    # J_new = J + alpha * (dy - J * dq) * dq^T / (dq^T * dq)

    j11, j12 = J[0]
    j21, j22 = J[1]

    # Where J predicted the end effector would go
    predicted_du = j11 * dtheta1 + j12 * dtheta2
    predicted_dv = j21 * dtheta1 + j22 * dtheta2

    # How wrong the prediction was
    miss_u = du - predicted_du
    miss_v = dv - predicted_dv

    dq_squared = dtheta1**2 + dtheta2**2

    j11 = j11 + ALPHA * miss_u * dtheta1 / dq_squared
    j12 = j12 + ALPHA * miss_u * dtheta2 / dq_squared
    j21 = j21 + ALPHA * miss_v * dtheta1 / dq_squared
    j22 = j22 + ALPHA * miss_v * dtheta2 / dq_squared

    return [[j11, j12],
            [j21, j22]]


# ---------------- 2.4: motion update ----------------

def compute_joint_step(J, error_u, error_v):

    j11, j12 = J[0]
    j21, j22 = J[1]

    determinant = j11 * j22 - j12 * j21

    # A determinant close to zero means the Jacobian cannot be safely inverted
    if abs(determinant) < 0.000001:
        print("Jacobian is singular.")
        return None

    # Solve J * dq = error  (same 2x2 inverse as numerical IK in lab 2)
    dtheta1 = (j22 * error_u - j12 * error_v) / determinant
    dtheta2 = (-j21 * error_u + j11 * error_v) / determinant

    # Take only part of the step: J is only accurate for small moves
    dtheta1 = dtheta1 * GAIN
    dtheta2 = dtheta2 * GAIN

    # Limit the step size, scaling both joints the same way so the direction stays the same
    largest = max(abs(dtheta1), abs(dtheta2))
    if largest > MAX_STEP:
        dtheta1 = dtheta1 * MAX_STEP / largest
        dtheta2 = dtheta2 * MAX_STEP / largest

    return dtheta1, dtheta2


# ---------------- 2.5 / 2.6: visual servoing ----------------

def visual_servo(moving_target):
    global stop_requested
    stop_requested = False

    if moving_target:
        print("\n================ VISUAL SERVOING: MOVING TARGET ================\n")
    else:
        print("\n================ VISUAL SERVOING: STATIONARY TARGET ================\n")


    # Save the stationary target before moving the arm
    if not moving_target:
        target = read_target()
        if target is None:
            return
        print("Stationary target saved:", target)
    else:
        target = None

    J = estimate_initial_jacobian()
    if J is None:
        return

    if moving_target:
        target = read_target()
        if target is None:
            return
    # J = estimate_initial_jacobian()
    # if J is None:
    #     return

    # # Measure the target now, while the arm is still away from it.
    # # When the end effector is over the dot it covers it, and a half-covered dot gives a wrong centre.
    # target = read_target()
    # if target is None:
    #     return

    print("\nTarget at u = {:.1f}, v = {:.1f} px\n".format(target[0], target[1]))

    previous_end_effector = None
    previous_step = None
    end_effector = None
    iteration = 0

    while True:

        end_effector, seen_target = measure_markers()

        if stop_requested:
            print("Stopped by user.")
            break

        if end_effector is None:
            print("Lost sight of the end effector, looking again...")
            previous_step = None
            continue

        # 2.6: follow the target, but only trust it when the arm is not covering it
        if moving_target and seen_target is not None:
            if distance(seen_target, end_effector) > OCCLUSION_RADIUS:
                target = seen_target

        # 2.3: correct J with what the last move actually did
        if previous_step is not None:
            du = end_effector[0] - previous_end_effector[0]
            dv = end_effector[1] - previous_end_effector[1]

            # If the arm barely moved (e.g. stuck at a joint limit), don't learn from it
            if math.sqrt(du**2 + dv**2) > MIN_PIXEL_CHANGE:
                dtheta1, dtheta2 = previous_step
                J = broyden_update(J, du, dv, dtheta1, dtheta2)

        # 2.2: stopping criterion
        error_u = target[0] - end_effector[0]
        error_v = target[1] - end_effector[1]
        error = math.sqrt(error_u**2 + error_v**2)

        print("Iteration {}: error = {:.1f} px".format(iteration, error))

        if error < TOLERANCE:
            if not moving_target:
                print("Reached the target.")
                break

            # Moving target: stay still and keep watching
            previous_step = None
            continue

        if not moving_target and iteration >= MAX_ITERATIONS:
            print("Did not reach the target in {} iterations.".format(MAX_ITERATIONS))
            break

        # 2.4 + 2.5: compute the joint move and send it to the brick
        step = compute_joint_step(J, error_u, error_v)
        if step is None:
            break

        dtheta1, dtheta2 = step
        move_joints(dtheta1, dtheta2)
        iteration += 1

        previous_end_effector = end_effector
        previous_step = (dtheta1, dtheta2)

    print("\n================ RESULT ================\n")
    print("Final Jacobian:")
    print_jacobian(J)

    if end_effector is not None:
        print("\nTarget position:       u = {:.1f}, v = {:.1f} px".format(target[0], target[1]))
        print("End-effector position: u = {:.1f}, v = {:.1f} px".format(end_effector[0], end_effector[1]))
        print("Positioning error:     {:.1f} px after {} iterations\n".format(
            distance(target, end_effector), iteration))


def main():
    global camera, brick

    print("\n================ INITIALIZATION ================\n")

    camera = cv2.VideoCapture(0, cv2.CAP_DC1394)   # firewire camera; use cv2.VideoCapture(0) for a webcam
    if not camera.isOpened():
        print("Could not open the camera.")
        return

    brick = connect_to_brick()

    try:
        while True:
            print("\n\n================================================")
            print("================ MAIN MENU =====================")
            print("================================================\n")

            print("s: Stationary target")
            print("m: Moving target")
            print("q: Quit")
            print()

            choice = input("Select an action : ").lower()

            if choice == 's':
                visual_servo(moving_target=False)

            elif choice == 'm':
                visual_servo(moving_target=True)

            elif choice == 'q':
                print("\n================ EXIT ================\n")
                brick.send("EXIT".encode("UTF-8"))
                break

            else:
                print("\n================ INVALID INPUT ================\n")
                print("Invalid selection. Please choose a valid menu option.\n")

    finally:
        brick.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()