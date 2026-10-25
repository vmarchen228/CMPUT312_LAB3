#!/usr/bin/python
# RUN ON LAPTOP USING PYTHON 3.6
#
# Lab 3, Part 2: Uncalibrated Visual Servoing (2.1 - 2.6).
# Start this first, then client.py on the brick.
# The brick moves each joint BY the received angles (relative, degrees) and replies
# "DONE", or "ERROR <reason>" if the move failed (e.g. target out of reach).
# Press ESC in the camera window to stop a run and go back to the menu.

import socket
import time
import threading
from collections import deque
from queue import Queue

import cv2
import numpy as np

# ---------------- tracking settings ----------------
# Tune these with color_tracking_hsv.py until only the sticker / marker shows up in the mask.
END_EFFECTOR_HSV_LOW = (100, 150, 0)      # blue sticker on the end effector
END_EFFECTOR_HSV_HIGH = (140, 255, 255)
TARGET_HSV_LOW = (0, 125, 50)             # red target marker
TARGET_HSV_HIGH = (10, 255, 255)
MIN_MARKER_AREA = 30       # pixels; smaller blobs are treated as noise
SAMPLES = 3                # recent frames averaged for each measurement
ESC_KEY = 27

# ---------------- servoing settings ----------------
PROBE_DEG = 10.0           # 2.1: test move per joint for the initial Jacobian
SETTLE_S = 0.3             # wait after a move so the arm stops shaking before the camera looks
TOLERANCE_PX = 5.0         # 2.2: stop when the end effector is this close to the target (pixels)
MAX_ITERATIONS = 50        # 2.2: give up on a stationary target after this many iterations
BROYDEN_ALPHA = 0.5        # 2.3: how strongly each observed move corrects the Jacobian
MIN_STEP_FOR_UPDATE = 0.5  # 2.3: skip the update for moves smaller than this (deg); pixel noise dominates
GAIN = 0.5                 # 2.4: fraction of the full step taken each iteration
MAX_STEP_DEG = 15.0        # 2.4: largest joint move per iteration, for safety
MAX_FAILED_MOVES = 3       # stationary target: give up after this many failed moves in a row (out of reach)


# This class handles the Server side of the comunication between the laptop and the brick.
class Server:
    def __init__(self, host, port):
       # setup server socket
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # We need to use the ip address that shows up in ipconfig for the usb ethernet adapter that handles the comunication between the PC and the brick
        print("Setting up Server\nAddress: " + host + "\nPort: " + str(port))

        serversocket.bind((host, port))
        # queue up to 5 requests
        serversocket.listen(5)
        self.cs, addr = serversocket.accept()
        print ("Connected to: " + str(addr))

    # Sends set of angles to the brick via TCP.
    # Input: base_angle [Float]: The angle by which we want the base to move
    #        joint_angle [Float]: The angle by which we want to joint to move
    #        queue [Thread-safe Queue]: Mutable data structure to store (and return) the messages received from the client
    def sendAngles(self, base_angle, joint_angle, queue):
        # Format in which the client expects the data: "angle1,angle2"
        data = str(base_angle) + "," + str(joint_angle)
        print("Sending Data: (" + data + ") to robot.")
        self.cs.send(data.encode("UTF-8"))
        # Waiting for the client (ev3 brick) to let the server know that it is done moving
        reply = self.cs.recv(128).decode("UTF-8")
        queue.put(reply)

    # Sends a termination message to the client. This will cause the client to exit "cleanly", after stopping the motors.
    def sendTermination(self):
        self.cs.send("EXIT".encode("UTF-8"))

    # Lets the client know that it should enable safety mode on its end
    def sendEnableSafetyMode(self):
        self.cs.send("SAFETY_ON".encode("UTF-8"))

    # Lets the client know that it should disable safety mode on its end
    def sendDisableSafetyMode(self):
        self.cs.send("SAFETY_OFF".encode("UTF-8"))


# ---------------- camera ----------------

def find_marker(frame, hsv_low, hsv_high):
    """Return the (u, v) pixel centre of the biggest blob in the colour range, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    mask = cv2.erode(mask, None, iterations=1)    # remove single-pixel noise
    mask = cv2.dilate(mask, None, iterations=2)   # fill small holes in the sticker

    # findContours returns 2 values in OpenCV 4 and 3 values in OpenCV 3;
    # the contour list is second-to-last in both.
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(biggest)
    if moments["m00"] < MIN_MARKER_AREA:
        return None
    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]
    return u, v


def draw_point(frame, point, color):
    if point is not None:
        u, v = point
        cv2.circle(frame, (int(u), int(v)), 10, color, 2)


class ContinuousCamera:
    """Capture, detect markers, and update preview independently of motor commands.

    Only this thread accesses VideoCapture and OpenCV preview functions.
    Other threads read copies of the latest averaged marker coordinates.
    """
    def __init__(self, device=0, backend=cv2.CAP_DC1394):
        self.capture = cv2.VideoCapture(device, backend)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open the camera")
        self.lock = threading.Lock()
        self.running = True
        self.esc_pressed = False
        self.failure = None
        self.points = deque(maxlen=SAMPLES)
        self.targets = deque(maxlen=SAMPLES)
        self.latest_point = None
        self.latest_target = None
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()

    def _camera_loop(self):
        try:
            cv2.namedWindow("Camera")
            while self.running:
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the camera")

                point = find_marker(frame, END_EFFECTOR_HSV_LOW, END_EFFECTOR_HSV_HIGH)
                target = find_marker(frame, TARGET_HSV_LOW, TARGET_HSV_HIGH)
                with self.lock:
                    # Ignore previously detected positions when marker is missing.
                    if point is None:
                        self.points.clear()
                    else:
                        self.points.append(point)
                    if target is None:
                        self.targets.clear()
                    else:
                        self.targets.append(target)
                    self.latest_point = (np.mean(self.points, axis=0).copy()
                                         if self.points else None)
                    self.latest_target = (np.mean(self.targets, axis=0).copy()
                                          if self.targets else None)
                    display_point = (self.latest_point.copy()
                                     if self.latest_point is not None else None)
                    display_target = (self.latest_target.copy()
                                      if self.latest_target is not None else None)

                draw_point(frame, display_point, (255, 0, 0))
                draw_point(frame, display_target, (0, 0, 255))
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ESC_KEY:
                    with self.lock:
                        self.esc_pressed = True
        except Exception as exc:
            with self.lock:
                self.failure = str(exc)
        finally:
            self.capture.release()
            cv2.destroyAllWindows()

    def read(self):
        """Return latest (end_effector, target, ESC); no camera I/O delay."""
        with self.lock:
            if self.failure is not None:
                raise RuntimeError(self.failure)
            point = self.latest_point.copy() if self.latest_point is not None else None
            target = self.latest_target.copy() if self.latest_target is not None else None
            esc = self.esc_pressed
            self.esc_pressed = False
        return point, target, ESC_KEY if esc else -1

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)


def read_camera(camera):
    """Read latest tracked positions without blocking the camera preview."""
    return camera.read()


def read_end_effector(camera):
    """Wait for a new visible end-effector position, not for camera capture."""
    for attempt in range(40):
        end_effector, target, key = read_camera(camera)
        if key == ESC_KEY:
            raise RuntimeError("Stopped by user")
        if end_effector is not None:
            return end_effector
        time.sleep(0.05)
    raise RuntimeError("End-effector sticker not found; check the HSV range and lighting")


# ---------------- talking to the brick ----------------

def move_joints(server, d_theta1, d_theta2):
    """Move each joint by a relative angle (degrees) and wait for the brick's reply.
    Returns True if the brick says "DONE", False if it reports an error."""
    server.sendAngles(d_theta1, d_theta2, queue)
    reply = queue.get()
    time.sleep(SETTLE_S)

    if reply == "":
        raise ConnectionError("The brick disconnected")
    if reply == "RESET" or reply.startswith("ERROR"):
        print("Brick could not do that move: " + reply)
        return False
    if reply != "DONE":
        raise RuntimeError("Unexpected EV3 reply: " + repr(reply))
    return True


# ---------------- 2.1: initial Jacobian ----------------

def estimate_initial_jacobian(server, camera):
    """Finite differences: move one joint at a time by PROBE_DEG and measure (du, dv).

        J = [ du/dtheta1   du/dtheta2 ]
            [ dv/dtheta1   dv/dtheta2 ]      units: pixels per degree

    Column 0 comes from moving theta1, column 1 from moving theta2."""
    J = np.zeros((2, 2))

    for joint in (0, 1):
        probe = [0.0, 0.0]
        probe[joint] = PROBE_DEG

        before = read_end_effector(camera)
        moved = move_joints(server, probe[0], probe[1])
        after = read_end_effector(camera)

        # undo the probe so both columns are measured from the same arm pose
        returned = move_joints(server, -probe[0], -probe[1])

        if not moved or not returned:
            raise RuntimeError("theta{} probe move failed; move the arm away from its joint limits and try again".format(joint + 1))

        J[:, joint] = (after - before) / PROBE_DEG

    print("\nInitial Jacobian (pixels per degree):")
    print_matrix(J)
    return J


def print_matrix(J):
    print("  [ {:8.3f}  {:8.3f} ]".format(J[0, 0], J[0, 1]))
    print("  [ {:8.3f}  {:8.3f} ]".format(J[1, 0], J[1, 1]))


# ---------------- 2.3: Broyden update ----------------

def broyden_update(J, dy, dq):
    """J_new = J + alpha * (dy - J dq) dq^T / (dq^T dq)

    dy: observed pixel change of the end effector, dq: joint change that caused it.
    Corrects J only in the direction we actually moved."""
    dq_squared = np.dot(dq, dq)
    if dq_squared < MIN_STEP_FOR_UPDATE**2:
        return J
    prediction_error = dy - J.dot(dq)
    return J + BROYDEN_ALPHA * np.outer(prediction_error, dq) / dq_squared


# ---------------- 2.4: motion update ----------------

def compute_joint_step(J, error):
    """Solve J * dq = error for the joint move dq (degrees) that should close the pixel error.
    Take only GAIN of it (the model is only locally linear) and cap the biggest joint move."""
    step = GAIN * np.linalg.pinv(J).dot(error)   # pinv instead of inv so a near-singular J doesn't blow up
    largest = np.max(np.abs(step))
    if largest > MAX_STEP_DEG:
        step = step * (MAX_STEP_DEG / largest)
    return step


# ---------------- 2.5 / 2.6: control loop ----------------

def visual_servo(server, camera, moving_target):
    """Stationary target: stop once within TOLERANCE_PX (2.2) and print the result.
    Moving target (2.6): never stop on its own; hold still while close, follow when the target moves.
    ESC in the camera window stops either mode."""
    J = estimate_initial_jacobian(server, camera)

    previous_end_effector = None
    previous_step = None
    end_effector = None
    target = None
    distance = None
    iteration = 0
    failed_moves = 0

    while True:
        end_effector, target, key = read_camera(camera)
        if key == ESC_KEY:
            print("Stopped by user.")
            break

        if end_effector is None or target is None:
            time.sleep(0.03)
            print("Lost sight of the end effector or the target; looking again...")
            previous_step = None
            continue

        # 2.3: correct J with what the last move actually did
        if previous_step is not None:
            observed_change = end_effector - previous_end_effector
            J = broyden_update(J, observed_change, previous_step)

        # 2.2: stopping criterion on the image-space error
        error = target - end_effector
        distance = np.sqrt(error[0]**2 + error[1]**2)
        print("Iteration {}: error = {:.1f} px".format(iteration, distance))

        if distance < TOLERANCE_PX:
            if not moving_target:
                print("Reached the target.")
                break
            previous_step = None      # moving target: hold still and keep watching
            time.sleep(0.03)
            continue

        if not moving_target and iteration >= MAX_ITERATIONS:
            print("Did not converge in {} iterations.".format(MAX_ITERATIONS))
            break

        # 2.4 + 2.5: compute the joint move and send it to the brick
        step = compute_joint_step(J, error)
        moved = move_joints(server, step[0], step[1])
        iteration += 1

        if moved:
            failed_moves = 0
            previous_end_effector = end_effector
            previous_step = step
        else:
            # the arm only did part of the move, so don't let it corrupt the Jacobian
            failed_moves += 1
            previous_step = None
            if not moving_target and failed_moves >= MAX_FAILED_MOVES:
                print("Target seems out of reach ({} failed moves in a row).".format(failed_moves))
                break

    print("\nFinal Jacobian:")
    print_matrix(J)
    if end_effector is not None and target is not None:
        print("Target position:       u = {:.1f}, v = {:.1f} px".format(target[0], target[1]))
        print("End-effector position: u = {:.1f}, v = {:.1f} px".format(end_effector[0], end_effector[1]))
        print("Positioning error:     {:.1f} px after {} iterations\n".format(distance, iteration))


host = "192.168.0.2"  # Change to laptop IP, if necessary
port = 9999

# Keep the same FireWire backend as the smooth color_tracking.py preview.
camera = ContinuousCamera(device=0, backend=cv2.CAP_DC1394)
queue = Queue()
server = None

try:
    server = Server(host, port)
    while True:
        user_input = input("Enter s for stationary target, or m for moving target (or q): ").lower()

        if user_input == "q":
            server.sendTermination()
            break
        if user_input not in ("s", "m"):
            print("Invalid command. Please try again.")
            continue

        try:
            visual_servo(server, camera, moving_target=(user_input == "m"))
        except RuntimeError as error:
            print("Run stopped: {}\n".format(error))
finally:
    camera.stop()
