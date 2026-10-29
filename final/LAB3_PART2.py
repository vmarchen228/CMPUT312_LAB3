#!/usr/bin/env python3
# LAB 3
import socket
import time
import math
import cv2

HOST = "192.168.0.2"
PORT = 9999

END_EFFECTOR_LOW = (100, 150, 0)
END_EFFECTOR_HIGH = (140, 255, 255)
TARGET_LOW = (0, 125, 50)
TARGET_HIGH = (10, 255, 255)
MIN_MARKER_AREA = 30

FLUSH_FRAMES = 5
SAMPLES = 5
ESC_KEY = 27

PROBE_ANGLE = 10.0
TOLERANCE = 10.0
MAX_ITERATIONS = 50

# How strongly each move corrects the Jacobian (Broyden)
ALPHA = 0.5
MIN_PIXEL_CHANGE = 2.0

# Take only part of each computed move, since J is accurate only for small moves
GAIN = 0.5
MAX_STEP = 15.0

# Ignore target readings this close to the sticker: the arm may be covering the dot
OCCLUSION_RADIUS = 40.0

camera = None
brick = None
stop_requested = False


def find_marker(hsv, low, high):
    # Erode removes noisy pixels, dilate fills small holes in the sticker
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    # [-2] works in both OpenCV 3 and 4 (they return different tuple sizes)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) == 0:
        return None

    biggest_contour = None
    biggest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if biggest_contour is None or area > biggest_area:
            biggest_contour = contour
            biggest_area = area

    # Moments: m00 = area, m10 = sum of u, m01 = sum of v
    moments = cv2.moments(biggest_contour)
    if moments["m00"] < MIN_MARKER_AREA:
        return None

    # Centre = average pixel position
    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]

    return u, v


def draw_marker(frame, point, color):
    if point is None:
        return

    center = (int(point[0]), int(point[1]))
    cv2.circle(frame, center, 10, color, 2)


def read_frame():
    global stop_requested

    ok, frame = camera.read()
    if not ok:
        return None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    end_effector = find_marker(hsv, END_EFFECTOR_LOW, END_EFFECTOR_HIGH)
    target = find_marker(hsv, TARGET_LOW, TARGET_HIGH)

    # OpenCV uses BGR order: (255, 0, 0) is blue
    draw_marker(frame, end_effector, (255, 0, 0))
    draw_marker(frame, target, (0, 0, 255))
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == ESC_KEY:
        stop_requested = True

    return end_effector, target


def average(points):
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

    du = point1[0] - point2[0]
    dv = point1[1] - point2[1]

    return math.sqrt(du**2 + dv**2)


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

    message = "{},{}".format(theta1_delta, theta2_delta)
    brick.send(message.encode("UTF-8"))

    # Poll for the reply every 1 ms and keep updating the camera,
    # so the video does not freeze while the arm moves
    brick.settimeout(0.001)

    reply = None
    while reply is None:
        try:
            reply = brick.recv(128).decode("UTF-8")
        except socket.timeout:
            read_frame()

    brick.settimeout(None)

    # Empty reply means the brick closed the connection
    if reply == "":
        raise SystemExit("The brick disconnected.")

    # Not "DONE" means a motor stopped a bit off target.
    # The camera will see where the arm really is, so continue
    if reply != "DONE":
        print("Warning from the brick: " + reply)


def probe_joint(theta1_delta, theta2_delta):
    before = read_end_effector()
    move_joints(theta1_delta, theta2_delta)
    after = read_end_effector()

    # Move back so both probes start from the same pose
    move_joints(-theta1_delta, -theta2_delta)

    if before is None or after is None:
        return None

    du = after[0] - before[0]
    dv = after[1] - before[1]

    return du, dv


def estimate_initial_jacobian():
    # J = [ du/dtheta1  du/dtheta2 ]   (pixels per degree)
    #     [ dv/dtheta1  dv/dtheta2 ]
    # Measured by turning one joint at a time and watching how the sticker moves

    print("\n================ ESTIMATE INITIAL JACOBIAN ================\n")

    # Joint 1 only -> first column, joint 2 only -> second column
    result1 = probe_joint(PROBE_ANGLE, 0.0)
    result2 = probe_joint(0.0, PROBE_ANGLE)

    if result1 is None or result2 is None:
        return None

    du1 = result1[0]
    dv1 = result1[1]
    du2 = result2[0]
    dv2 = result2[1]

    probe1_pixels = math.sqrt(du1**2 + dv1**2)
    probe2_pixels = math.sqrt(du2**2 + dv2**2)

    # A tiny movement means the joint was likely blocked at a limit,
    # so that column of J would be just noise
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


def broyden_update(J, du, dv, dtheta1, dtheta2):
    # J changes as the arm moves, so after each move we correct it
    # with what actually happened:
    #   J_new = J + alpha * (dy - J*dq) * dq^T / (dq^T * dq)
    #   dy = observed pixel move, dq = joint move

    j11 = J[0][0]
    j12 = J[0][1]
    j21 = J[1][0]
    j22 = J[1][1]

    # Pixel move that J predicted (J * dq)
    predicted_du = j11 * dtheta1 + j12 * dtheta2
    predicted_dv = j21 * dtheta1 + j22 * dtheta2

    # Prediction error (dy - J * dq)
    miss_u = du - predicted_du
    miss_v = dv - predicted_dv

    # dq^T * dq normalizes the correction so it does not depend on the move size
    dq_squared = dtheta1**2 + dtheta2**2

    # Each entry is corrected in proportion to how much its joint moved
    j11 = j11 + ALPHA * miss_u * dtheta1 / dq_squared
    j12 = j12 + ALPHA * miss_u * dtheta2 / dq_squared
    j21 = j21 + ALPHA * miss_v * dtheta1 / dq_squared
    j22 = j22 + ALPHA * miss_v * dtheta2 / dq_squared

    J_new = [[j11, j12],
             [j21, j22]]

    return J_new


def compute_joint_step(J, error_u, error_v):
    # Solve J * dq = pixel error for dq

    j11 = J[0][0]
    j12 = J[0][1]
    j21 = J[1][0]
    j22 = J[1][1]

    determinant = j11 * j22 - j12 * j21

    # A determinant close to zero means J cannot be inverted
    if abs(determinant) < 0.000001:
        print("Jacobian is singular.")
        return None

    # 2x2 inverse: inv([a b; c d]) = 
    #                                  [d -b  /  (a*d - b*c)
    #                                   -c a]      
    # Same as numerical IK in lab 2
    dtheta1 = (j22 * error_u - j12 * error_v) / determinant
    dtheta2 = (-j21 * error_u + j11 * error_v) / determinant

    dtheta1 = dtheta1 * GAIN
    dtheta2 = dtheta2 * GAIN

    # If either joint exceeds MAX_STEP, scale both down equally
    # so the direction of the move stays the same
    largest = max(abs(dtheta1), abs(dtheta2))
    if largest > MAX_STEP:
        scale = MAX_STEP / largest
        dtheta1 = dtheta1 * scale
        dtheta2 = dtheta2 * scale

    return dtheta1, dtheta2


def visual_servo(moving_target):
    global stop_requested
    stop_requested = False

    if moving_target:
        print("\n================== VISUAL SERVOING: MOVING TARGET ==================\n")
    else:
        print("\n================ VISUAL SERVOING: STATIONARY TARGET ================\n")

    target = None

    # Stationary target: save it before the arm moves,
    # because the sticker covers the dot when it gets there
    if not moving_target:
        target = read_target()
        if target is None:
            return
        print("Stationary target saved:", target)

    J = estimate_initial_jacobian()
    if J is None:
        return

    if moving_target:
        target = read_target()
        if target is None:
            return

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

        # Can't see the sticker: measure again, and don't learn from the last move
        if end_effector is None:
            print("Lost sight of the end effector, looking again...")
            previous_step = None
            continue

        # Update a moving target only when it is far from the stickers
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

        error_u = target[0] - end_effector[0]
        error_v = target[1] - end_effector[1]
        error = math.sqrt(error_u**2 + error_v**2)

        print(
            "Iteration {}: error = {:.1f} px"
            .format(iteration, error)
        )

        if error < TOLERANCE:
            if not moving_target:
                print("Reached the target.")
                break

            # Moving target: hold still and keep watching
            previous_step = None
            continue

        if not moving_target and iteration >= MAX_ITERATIONS:
            print(
                "Did not reach the target in {} iterations."
                .format(MAX_ITERATIONS)
            )
            break

        step = compute_joint_step(J, error_u, error_v)
        if step is None:
            break

        dtheta1 = step[0]
        dtheta2 = step[1]

        move_joints(dtheta1, dtheta2)
        iteration = iteration + 1

        previous_end_effector = end_effector
        previous_step = (dtheta1, dtheta2)

    print("\n================ RESULT ================\n")
    print("Final Jacobian:")
    print_jacobian(J)

    if end_effector is not None:
        final_error = distance(target, end_effector)

        print(
            "\nTarget position:       u = {:.1f}, v = {:.1f} px"
            .format(target[0], target[1])
        )

        print(
            "End-effector position: u = {:.1f}, v = {:.1f} px"
            .format(end_effector[0], end_effector[1])
        )

        print(
            "Positioning error:     {:.1f} px after {} iterations\n"
            .format(final_error, iteration)
        )


def main():
    global camera, brick

    print("\n================ INITIALIZATION ================\n")

    # Lab FireWire camera; use cv2.VideoCapture(0) for a webcam
    camera = cv2.VideoCapture(0, cv2.CAP_DC1394)
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

            choice = input("\nSelect an action : ").lower()

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