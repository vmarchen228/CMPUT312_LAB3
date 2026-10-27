#!/usr/bin/env python3
# CMPUT 312 Lab 3: uncalibrated visual servoing (laptop side).
# Start this first, then client.py on the brick.
# Sends relative joint moves "dtheta1,dtheta2" to the brick; the brick replies "DONE" (or "RESET"/"ERROR" if a motor stopped off target).
# Press ESC in the camera window to stop a run.

import socket
import time
import math
import cv2


# ---------------- settings ----------------

HOST = "192.168.0.2"    # laptop's IP on the USB link to the brick
PORT = 9999

# HSV colour ranges (tune with color_tracking_hsv.py)
END_EFFECTOR_LOW = (100, 150, 0)      # blue sticker on the end effector
END_EFFECTOR_HIGH = (140, 255, 255)
TARGET_LOW = (0, 125, 50)             # red target dot
TARGET_HIGH = (10, 255, 255)
MIN_MARKER_AREA = 30                  # blobs smaller than this (pixels) are noise

FLUSH_FRAMES = 5        # old frames skipped so we never measure while the arm is still moving
SAMPLES = 5             # frames averaged per measurement to reduce jitter
ESC_KEY = 27

PROBE_ANGLE = 10.0      # degrees each joint turns to measure the starting Jacobian
TOLERANCE = 10.0        # stop when the sticker is this close to the target (pixels)
MAX_ITERATIONS = 50     # give up on a stationary target after this many moves
ALPHA = 0.5             # how strongly each move corrects the Jacobian
MIN_PIXEL_CHANGE = 2.0  # don't correct the Jacobian from moves smaller than this (pixels): mostly noise
GAIN = 0.5              # take half the computed move, since the Jacobian is only accurate for small movesx
MAX_STEP = 15.0         # largest joint move per step (degrees), for safety
OCCLUSION_RADIUS = 40.0 # ignore target readings this close to the sticker: the arm may be covering the dot

camera = None           # set in main()
brick = None            # set in main()
stop_requested = False  # set True when ESC is pressed


# ---------------- camera ----------------

def find_marker(hsv, low, high):
    """Return the (u, v) centre of the biggest blob in the colour range, or None."""


    #Create a mask for the colour range, then erode and dilate to remove noise and fill holes
    mask = cv2.inRange(hsv, low, high)            # white where the colour is in range, black elsewhere
    mask = cv2.erode(mask, None, iterations=1)    # shrink white areas: removes single noisy pixels
    mask = cv2.dilate(mask, None, iterations=2)   # grow them back: fills small holes in the sticker

    # Outlines of all white areas. [-2] works in both OpenCV 3 and 4 (they return different tuple sizes).
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) == 0:
        return None

    # Keep the largest blob: most likely the real marker, not a reflection
    biggest_contour = None
    biggest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if biggest_contour is None or area > biggest_area:
            biggest_contour = contour
            biggest_area = area

    # Moments: m00 = area, m10 = sum of u, m01 = sum of v
    moments = cv2.moments(biggest_contour)
    #Reject area that is small
    if moments["m00"] < MIN_MARKER_AREA:
        return None

    # Centre = average pixel position
    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]

    return u, v


def draw_marker(frame, point, color):
    """Circle a found marker on screen."""

    if point is None:
        return

    center = (int(point[0]), int(point[1]))     # cv2.circle needs whole pixels
    cv2.circle(frame, center, 10, color, 2)     # radius 10 px, line 2 px


def read_frame():
    """Grab one frame, find both markers, show it, check ESC. Returns (end_effector, target)."""

    global stop_requested

    ok, frame = camera.read()
    if not ok:
        return None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # HSV separates colour from brightness, easier to threshold

    end_effector = find_marker(hsv, END_EFFECTOR_LOW, END_EFFECTOR_HIGH)
    target = find_marker(hsv, TARGET_LOW, TARGET_HIGH)

    draw_marker(frame, end_effector, (255, 0, 0))   # blue circle (OpenCV uses BGR order)
    draw_marker(frame, target, (0, 0, 255))         # red circle
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)    # redraw window, wait 1 ms for a key
    # key = key & 0xFF        # keep only the low 8 bits (some systems add extra bits)

    if key == ESC_KEY:
        stop_requested = True

    return end_effector, target


def average(points):
    """Average (u, v) of a list of points, or None if empty."""

    if len(points) == 0:
        return None

    total_u = 0.0
    total_v = 0.0
    for point in points:
        total_u = total_u + point[0]
        total_v = total_v + point[1]

    u = total_u / len(points)
    v = total_v / len(points)

    return u, v


def measure_markers():
    """Skip old frames, then average several new ones. Returns (end_effector, target)."""

    # Throw away buffered frames that may show the arm mid-move
    for frame_index in range(FLUSH_FRAMES):
        read_frame()

    end_effector_points = []
    target_points = []

    for frame_index in range(SAMPLES):
        end_effector, target = read_frame()

        if end_effector is not None:
            end_effector_points.append(end_effector)

        if target is not None:
            target_points.append(target)

    end_effector_average = average(end_effector_points)
    target_average = average(target_points)

    return end_effector_average, target_average


def read_end_effector():
    """Measure until the sticker is found (up to 20 tries)."""

    for attempt in range(20):
        end_effector, target = measure_markers()
        if end_effector is not None:
            return end_effector

    print("End effector not found. Check the HSV range and lighting.")
    return None


def read_target():
    """Measure until the target dot is found (up to 20 tries)."""

    for attempt in range(20):
        end_effector, target = measure_markers()
        if target is not None:
            return target

    print("Target not found. Check the HSV range and lighting.")
    return None


def distance(point1, point2):
    """Pixel distance between two (u, v) points."""

    du = point1[0] - point2[0]
    dv = point1[1] - point2[1]

    return math.sqrt(du**2 + dv**2)


# ---------------- communication with the brick ----------------

def connect_to_brick():
    """Wait for the brick to connect over TCP. Returns the connection."""

    print("\n================ WAITING FOR THE BRICK ================\n")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)        # IPv4 TCP socket
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)      # allow restarting without "address already in use"
    listener.bind((HOST, PORT))
    listener.listen(1)

    print("Address: {}  Port: {}".format(HOST, PORT))
    print("Now run client.py on the brick...")

    connection, address = listener.accept()     # blocks until the brick connects
    print("Connected to: " + str(address))

    return connection


def move_joints(theta1_delta, theta2_delta):
    """Tell the brick to turn each joint BY the given degrees, and wait for its reply."""

    print("Moving joints by theta1 = {:.2f} deg, theta2 = {:.2f} deg".format(theta1_delta, theta2_delta))

    message = "{},{}".format(theta1_delta, theta2_delta)    # format the brick expects: "dtheta1,dtheta2"
    brick.send(message.encode("UTF-8"))

    # Poll for the reply every 1 ms and update the camera meanwhile, so the video doesn't freeze
    brick.settimeout(0.001)

    reply = None
    while reply is None:
        try:
            reply = brick.recv(128).decode("UTF-8")
        except socket.timeout:
            read_frame()

    brick.settimeout(None)      # back to normal blocking mode

    if reply == "":             # empty reply = brick closed the connection
        raise SystemExit("The brick disconnected.")

    # Not "DONE" = motor stopped a bit off target; the camera will see where it really is, so continue
    if reply != "DONE":
        print("Warning from the brick: " + reply)


# ---------------- starting Jacobian ----------------
# J = [ du/dtheta1  du/dtheta2 ]   (pixels per degree)
#     [ dv/dtheta1  dv/dtheta2 ]
# Measured by turning one joint at a time and watching how the sticker moves.

def probe_joint(theta1_delta, theta2_delta):
    """Make a test move, measure the sticker's pixel change, then undo the move. Returns (du, dv)."""

    before = read_end_effector()
    move_joints(theta1_delta, theta2_delta)
    after = read_end_effector()

    move_joints(-theta1_delta, -theta2_delta)   # move back so both probes start from the same pose

    if before is None or after is None:
        return None

    du = after[0] - before[0]
    dv = after[1] - before[1]

    return du, dv


def estimate_initial_jacobian():
    """Build J from two test moves. Returns J or None."""

    print("\n================ ESTIMATE INITIAL JACOBIAN ================\n")

    result1 = probe_joint(PROBE_ANGLE, 0.0)     # joint 1 only -> first column
    result2 = probe_joint(0.0, PROBE_ANGLE)     # joint 2 only -> second column

    if result1 is None or result2 is None:
        return None

    du1 = result1[0]
    dv1 = result1[1]
    du2 = result2[0]
    dv2 = result2[1]

    probe1_pixels = math.sqrt(du1**2 + dv1**2)
    probe2_pixels = math.sqrt(du2**2 + dv2**2)

    # A tiny movement means the joint was likely blocked at a limit; that column would be noise
    if probe1_pixels < 3 or probe2_pixels < 3:
        print("A probe barely moved the end effector. Move the arm away from its joint limits and try again.")
        return None

    # Pixels moved / degrees turned = pixels per degree
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


# ---------------- Broyden update ----------------
# J changes as the arm moves, so after each move we correct it with what actually happened:
#   J_new = J + alpha * (dy - J*dq) * dq^T / (dq^T * dq)
#   dy = observed pixel move, dq = joint move

def broyden_update(J, du, dv, dtheta1, dtheta2):
    """Return J corrected by the observed move."""

    j11 = J[0][0]
    j12 = J[0][1]
    j21 = J[1][0]
    j22 = J[1][1]

    # Pixel move J predicted (J * dq)
    predicted_du = j11 * dtheta1 + j12 * dtheta2
    predicted_dv = j21 * dtheta1 + j22 * dtheta2

    # Prediction error (dy - J * dq)
    miss_u = du - predicted_du
    miss_v = dv - predicted_dv

    # dq^T * dq: normalizes the correction so it doesn't depend on the move size
    dq_squared = dtheta1**2 + dtheta2**2

    # Each entry is corrected in proportion to how much its joint moved
    j11 = j11 + ALPHA * miss_u * dtheta1 / dq_squared
    j12 = j12 + ALPHA * miss_u * dtheta2 / dq_squared
    j21 = j21 + ALPHA * miss_v * dtheta1 / dq_squared
    j22 = j22 + ALPHA * miss_v * dtheta2 / dq_squared

    J_new = [[j11, j12],
             [j21, j22]]

    return J_new


# ---------------- next joint move ----------------
# Solve J * dq = pixel error for dq.

def compute_joint_step(J, error_u, error_v):
    """Return (dtheta1, dtheta2) that moves the sticker toward the target, or None if J is singular."""

    j11 = J[0][0]
    j12 = J[0][1]
    j21 = J[1][0]
    j22 = J[1][1]

    determinant = j11 * j22 - j12 * j21

    # Near-zero determinant: J can't be inverted (both joints move the sticker the same way)
    if abs(determinant) < 0.000001:
        print("Jacobian is singular.")
        return None

    # 2x2 inverse: inv([a b; c d]) = [d -b; -c a] / (a*d - b*c)   (same as numerical IK in lab 2)
    dtheta1 = (j22 * error_u - j12 * error_v) / determinant
    dtheta2 = (-j21 * error_u + j11 * error_v) / determinant

    # Take only part of the move: J is only accurate near the current pose
    dtheta1 = dtheta1 * GAIN
    dtheta2 = dtheta2 * GAIN

    # If either joint exceeds MAX_STEP, scale both down equally so the direction stays the same
    largest = max(abs(dtheta1), abs(dtheta2))
    if largest > MAX_STEP:
        scale = MAX_STEP / largest
        dtheta1 = dtheta1 * scale
        dtheta2 = dtheta2 * scale

    return dtheta1, dtheta2


# ---------------- control loop ----------------
# Each iteration: measure -> correct J with the last move -> stop if close enough -> compute and send next move.

def visual_servo(moving_target):
    """One visual-servoing run. moving_target=True keeps following the target instead of stopping."""

    global stop_requested
    stop_requested = False      # clear any ESC from a previous run

    if moving_target:
        print("\n================ VISUAL SERVOING: MOVING TARGET ================\n")
    else:
        print("\n================ VISUAL SERVOING: STATIONARY TARGET ================\n")

    target = None

    # Stationary target: save it before the arm moves, because the sticker covers the dot when it gets there
    if not moving_target:
        target = read_target()
        if target is None:
            return
        print("Stationary target saved:", target)

    J = estimate_initial_jacobian()
    if J is None:
        return

    # Moving target: first reading (updated inside the loop)
    if moving_target:
        target = read_target()
        if target is None:
            return

    print("\nTarget at u = {:.1f}, v = {:.1f} px\n".format(target[0], target[1]))

    previous_end_effector = None   # sticker position before the last move
    previous_step = None           # last joint move, or None if there's nothing to learn from
    end_effector = None
    iteration = 0

    while True:

        end_effector, seen_target = measure_markers()

        if stop_requested:
            print("Stopped by user.")
            break

        # Can't see the sticker: measure again, and don't learn from the last move
        if end_effector is None:
            print("Lost sight of the end effector, looking again...")
            previous_step = None
            continue

        # Update a moving target only when it's far from the sticker (not covered by the arm)
        if moving_target and seen_target is not None:
            if distance(seen_target, end_effector) > OCCLUSION_RADIUS:
                target = seen_target

        # Correct J with what the last move actually did
        if previous_step is not None:
            du = end_effector[0] - previous_end_effector[0]
            dv = end_effector[1] - previous_end_effector[1]
            pixels_moved = math.sqrt(du**2 + dv**2)

            # Skip tiny moves (e.g. stuck at a joint limit): mostly noise
            if pixels_moved > MIN_PIXEL_CHANGE:
                dtheta1 = previous_step[0]
                dtheta2 = previous_step[1]
                J = broyden_update(J, du, dv, dtheta1, dtheta2)

        # Remaining pixel error to the target
        error_u = target[0] - end_effector[0]
        error_v = target[1] - end_effector[1]
        error = math.sqrt(error_u**2 + error_v**2)

        print("Iteration {}: error = {:.1f} px".format(iteration, error))

        # Stopping criterion
        if error < TOLERANCE:
            if not moving_target:
                print("Reached the target.")
                break

            # Moving target: hold still and keep watching (no move, so nothing to learn next time)
            previous_step = None
            continue

        if not moving_target and iteration >= MAX_ITERATIONS:
            print("Did not reach the target in {} iterations.".format(MAX_ITERATIONS))
            break

        step = compute_joint_step(J, error_u, error_v)
        if step is None:
            break

        dtheta1 = step[0]
        dtheta2 = step[1]

        move_joints(dtheta1, dtheta2)
        iteration = iteration + 1

        # Remember this move so the next iteration can correct J with it
        previous_end_effector = end_effector
        previous_step = (dtheta1, dtheta2)

    print("\n================ RESULT ================\n")
    print("Final Jacobian:")
    print_jacobian(J)

    if end_effector is not None:
        final_error = distance(target, end_effector)
        print("\nTarget position:       u = {:.1f}, v = {:.1f} px".format(target[0], target[1]))
        print("End-effector position: u = {:.1f}, v = {:.1f} px".format(end_effector[0], end_effector[1]))
        print("Positioning error:     {:.1f} px after {} iterations\n".format(final_error, iteration))


# ---------------- main ----------------

def main():
    global camera, brick

    print("\n================ INITIALIZATION ================\n")

    camera = cv2.VideoCapture(0, cv2.CAP_DC1394)    # lab FireWire camera; use cv2.VideoCapture(0) for a webcam
    if not camera.isOpened():
        print("Could not open the camera.")
        return

    brick = connect_to_brick()

    # finally: always close the connection, camera and window, even on errors or Ctrl+C
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
                brick.send("EXIT".encode("UTF-8"))      # tells client.py to exit its loop
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