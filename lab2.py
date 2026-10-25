#LAB 2
#!/usr/bin/env python3
import time, math
from ev3dev2.motor import  LargeMotor, OUTPUT_A, OUTPUT_B, SpeedPercent
from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import TouchSensor

touch_sensor = TouchSensor(INPUT_1)

upper_motor = LargeMotor(OUTPUT_A) # L1 + L2; theta1
bottom_motor = LargeMotor(OUTPUT_B) # L2; theta2

upper_motor.ramp_up_sp = 1500
upper_motor.ramp_down_sp = 1500

bottom_motor.ramp_up_sp = 1500
bottom_motor.ramp_down_sp = 1500

L1 = 11.5
L2 = 9.25

THETA1_MIN = -180.0
THETA1_MAX = 180.0
THETA2_MIN = -180.0
THETA2_MAX = 180.0

MOTOR_SPEED = 9

NEWTON_MAX_ITERATIONS = 50
NEWTON_TOLERANCE = 0.01

def release_motors():
    print("\n================ RELEASE MOTORS ================\n")

    upper_motor.off(brake=False)
    bottom_motor.off(brake=False)

    print("Motors released\n")


def zero_arm():
    print("\n================ ZERO ARM ================\n")

    upper_motor.position = 0
    bottom_motor.position = 0

    print("Current arm position set as theta1 = 0 deg, theta2 = 0 deg\n")


def calibrate_limits():
    global THETA1_MIN, THETA1_MAX, THETA2_MIN, THETA2_MAX

    print("\n================ CALIBRATE JOINT LIMITS ================\n")

    release_motors()

    input("Move theta1 to one limit and press Enter...")
    theta1_a, theta2, x, y = current_position()

    print()

    input("Move theta1 to the other limit and press Enter...")
    theta1_b, theta2, x, y = current_position()

    THETA1_MIN = min(theta1_a, theta1_b)
    THETA1_MAX = max(theta1_a, theta1_b)

    print()

    input("Move theta2 to one limit and press Enter...")
    theta1, theta2_a, x, y = current_position()

    print()

    input("Move theta2 to the other limit and press Enter...")
    theta1, theta2_b, x, y = current_position()

    THETA2_MIN = min(theta2_a, theta2_b)
    THETA2_MAX = max(theta2_a, theta2_b)

    print("\n================ CALIBRATION COMPLETE ================\n")

    print("Theta1 range: {:.2f} to {:.2f} deg".format(
        THETA1_MIN, THETA1_MAX
    ))

    print("Theta2 range: {:.2f} to {:.2f} deg".format(
        THETA2_MIN, THETA2_MAX
    ))

    print()


def forward_kinematics(theta1_deg, theta2_deg):

    # Convert degrees to radians
    theta1 = math.radians(theta1_deg)
    theta2 = math.radians(theta2_deg)

    # Compute position of the end effector
    x = (L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2))
    y = (L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2))

    return x, y


def move_to_angles(theta1_deg_target, theta2_deg_target):

    print("\n================ MOVE TO ANGLES ================\n")

    if theta1_deg_target < THETA1_MIN or theta1_deg_target > THETA1_MAX:
        print("\nTheta 1 is out of range.\n")
        return

    if theta2_deg_target < THETA2_MIN or theta2_deg_target > THETA2_MAX:
        print("\nTheta 2 is out of range.\n")
        return
    
    # Gear ratio 40/24
    bottom_motor_angle = theta1_deg_target + (40.0 / 24.0) * theta2_deg_target

    upper_motor.on_to_position(
        SpeedPercent(MOTOR_SPEED),
        theta1_deg_target,
        brake=False,
        # block=False
    )

    bottom_motor.on_to_position(
        SpeedPercent(MOTOR_SPEED),
        bottom_motor_angle,
        brake=False,
        # block=False
    )

    time.sleep(0.5)

    theta1_deg, theta2_deg, x, y = current_position()

    return theta1_deg, theta2_deg, x, y


def measure_distance():
    print("\n================ MEASURE DISTANCE ================\n")

    release_motors()

    print("Move end effector to point 1 and press touch sensor...")
    x1, y1 = wait_for_press()

    print("\nMove end effector to point 2 and press touch sensor...")
    x2, y2 = wait_for_press()

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    print("\n================ DISTANCE RESULT ================\n")
    print("Distance: {:.2f} cm\n".format(distance))


def measure_angle():
    print("\n================ MEASURE ANGLE ================\n")

    release_motors()

    print("Move end effector to intersection point and press touch sensor...")
    x0, y0 = wait_for_press()

    print("\nMove end effector to point on the first line and press touch sensor...")
    x1, y1 = wait_for_press()

    print("\nMove end effector to point on the second line and press touch sensor...")
    x2, y2 = wait_for_press()

    vector1_x = x1 - x0
    vector1_y = y1 - y0

    vector2_x = x2 - x0
    vector2_y = y2 - y0

    numerator = vector1_x * vector2_x + vector1_y * vector2_y

    vector1_mag = math.sqrt(vector1_x**2 + vector1_y**2)
    vector2_mag = math.sqrt(vector2_x**2 + vector2_y**2)

    denominator = vector1_mag * vector2_mag
    if denominator <0.00001:

        print("Denominator is zero or close to 0")
        return


    cos_angle = numerator / denominator
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))

    print("\n================ ANGLE RESULT ================\n")
    print("Angle between lines: {:.2f} deg\n".format(angle))


def current_position():
    print("\n================ CURRENT POSITION ================\n")

    theta1 = upper_motor.position
    theta2 = (bottom_motor.position - theta1) * (24.0 / 40.0)
    x, y = forward_kinematics(theta1, theta2)

    print(
        "Encoder angles: theta1 = {:.2f} deg, theta2 = {:.2f} deg"
        .format(theta1, theta2)
    )

    print(
        "Encoder-estimated position: x = {:.2f} cm, y = {:.2f} cm"
        .format(x, y)
    )

    print()

    return theta1, theta2, x, y


def wait_for_press():
    while not touch_sensor.is_pressed:
        time.sleep(0.05)

    theta1, theta2, x, y = current_position()

    while touch_sensor.is_pressed:
        time.sleep(0.05)

    return x, y


def analytical_IK(x_target, y_target):

    cos_theta2 = (
        x_target**2 + y_target**2 - L1**2 - L2**2
    ) / (2.0 * L1 * L2)

    if cos_theta2 < -1.0 or cos_theta2 > 1.0:
        print("Position is outside the workspace.")
        return None

    theta2 = math.atan2(
        math.sqrt(1.0 - cos_theta2**2),
        cos_theta2
    )

    theta1 = math.atan2(y_target, x_target) - math.atan2(
        L2 * math.sin(theta2),
        L1 + L2 * math.cos(theta2)
    )

    return math.degrees(theta1), math.degrees(theta2)


def numerical_IK(x_target, y_target):

    distance_to_target = math.sqrt(x_target**2 + y_target**2)

    if distance_to_target > (L1 + L2):
        print("Position is outside the workspace.")
        return None

    # Check whether the target is too close to the base to be reached

    if distance_to_target < abs(L1 - L2):
        print("Position is outside the workspace.")
        return None

    theta1 = math.radians(upper_motor.position)
    theta2 = math.radians(
        (bottom_motor.position - upper_motor.position) * (24.0 / 40.0)
    )

    # Initial guess check
    if abs(math.sin(theta2)) < 0.1:
        theta1 = math.radians(20.0)
        theta2 = math.radians(20.0)

    print(
        "Initial guess: theta1 = {:.2f} deg, theta2 = {:.2f} deg"
        .format(math.degrees(theta1), math.degrees(theta2))
    )


    for i in range(NEWTON_MAX_ITERATIONS):

        # Forward kinematics:
        # calculate the current x,y-coordinate of the end effector
        x = (
            L1 * math.cos(theta1)
            + L2 * math.cos(theta1 + theta2)
        )

        y = (
            L1 * math.sin(theta1)
            + L2 * math.sin(theta1 + theta2)
        )
        
        
        
        # Calculate the error between the current position
        # and the desired target position
        error_x = x - x_target
        error_y = y - y_target


        # Calculate the total position error using Euclidean distance
        error = math.sqrt(error_x**2 + error_y**2)

        print(
            "Iteration {}: theta1 = {:.2f}, theta2 = {:.2f}, error = {:.4f}"
            .format(
                i,
                math.degrees(theta1),
                math.degrees(theta2),
                error
            )
        )
        # If the error is small enough, the IK solution has converged

        if error < NEWTON_TOLERANCE:
            print("Converged in {} iterations".format(i))
            theta1_deg = math.degrees(theta1)
            theta2_deg = math.degrees(theta2)

             # Normalize the angles so they remain between -180 and 180 degrees
            theta1_deg_norm = (theta1_deg + 180.0) % 360.0 - 180.0
            theta2_deg_norm = (theta2_deg + 180.0) % 360.0 - 180.0

            # Return the final inverse-kinematics solution
            return theta1_deg_norm, theta2_deg_norm

        # Calculate the Jacobian matrix.
        # The Jacobian describes how small changes in joint angles
        # affect the x and y position of the end effector.

        # dx / dtheta1

        j11 = (
            -L1 * math.sin(theta1)
            - L2 * math.sin(theta1 + theta2)
        )
        # dx / dtheta2


        j12 = -L2 * math.sin(theta1 + theta2)

        # dy / dtheta1

        j21 = (
            L1 * math.cos(theta1)
            + L2 * math.cos(theta1 + theta2)
        )

        # dy / dtheta2

        j22 = L2 * math.cos(theta1 + theta2)

        determinant = j11 * j22 - j12 * j21

        # A determinant close to zero means the Jacobian cannot
        # be safely inverted.
        if abs(determinant) < 0.000001:
            print("Jacobian is singular.")
            return None

        # Solve J * delta_theta = error to determine
        # how much theta1 and theta2 should change
        delta_theta1 = (
            j22 * error_x - j12 * error_y
        ) / determinant

        delta_theta2 = (
            -j21 * error_x + j11 * error_y
        ) / determinant

        theta1 = theta1 - delta_theta1 * 0.2
        theta2 = theta2 - delta_theta2 * 0.2

    print("Numerical IK did not converge.")
    return None


def move_to_position(x_target=None, y_target=None):
    print("\n================ MOVE TO POSITION USING IK ================\n")

    if x_target is None or y_target is None:
        while True:
            try:
                user_input = input(
                    "Enter target coordinates (x and y) separated by a space: "
                )

                x_target, y_target = map(float, user_input.split())

                print(
                    "Entered coordinates: x = {:.2f}, y = {:.2f}"
                    .format(x_target, y_target)
                )

                break

            except ValueError:
                print(
                    "Error: Please enter exactly two valid numbers separated by a space."
                )

    while True:
        choice = input("Enter a for analytical, n for numerical: ")

        if choice.lower() == 'a':

            result = analytical_IK(x_target, y_target)

            if result is None:
                return

            theta1, theta2 = result
            break

        elif choice.lower() == 'n':

            result = numerical_IK(x_target, y_target)

            if result is None:
                return

            theta1, theta2 = result
            break

        else:
            print("Invalid selection, please choose \"a\" or \"n\"")

    print(
        "\n\nRequested target position: x = {:.2f} cm, y = {:.2f} cm"
        .format(x_target, y_target)
    )

    print(
        "Requested solution type: {}"
        .format("analytical" if choice.lower() == 'a' else "numerical")
    )

    print(
        "Calculated joint angles: theta1 = {:.2f} deg, theta2 = {:.2f} deg"
        .format(theta1, theta2)
    )

    result = move_to_angles(theta1, theta2)

    if result is None:
        return

    theta1_deg, theta2_deg, x, y = result

    print(
        "Position error (Euclidean distance): {:.2f} cm"
        .format(math.sqrt((x - x_target)**2 + (y - y_target)**2))
    )

def midpoint():
    print("\n================ MIDPOINT ================\n")
    release_motors()

    print("Move end effector to point 1 and press touch sensor...")
    x1, y1 = wait_for_press()

    print("\nMove end effector to point 2 and press touch sensor...")
    x2, y2 = wait_for_press()

    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    print(
        "\nCalculated midpoint: x = {:.2f} cm, y = {:.2f} cm"
        .format(x_mid, y_mid)
    )

    move_to_position(x_mid, y_mid)




def main():
    print("\n================ ROBOT INITIALIZATION ================\n")

    input("Place arm in zero position and press Enter...")
    zero_arm()

    while True:
        print("\n\n\n================================================")
        print("================ MAIN MENU =====================")
        print("================================================\n")

        print("1: Move to angles")
        print("2: Measure distance")
        print("3: Measure angle")
        print("4: Position to angles")
        print("5: Midpoint")
        print()
        print("c: Calibrate joint limits")
        print("cp: Current position (angles)")
        print("r: Release motors")
        print("z: Zero angle")
        print("q: Quit")
        print()

        print("\nWARNING: Please make sure joint limits are calibrated before moving the arm.")
        print()

        choice = input("Select an action : ")

        if choice == '1':
            try:
                theta1_deg = float(input("Enter theta 1: ").strip())
                theta2_deg = float(input("Enter theta 2: ").strip())
            except ValueError:
                print("Error: Please enter valid numbers.")
                continue

            print(
                "\nCommanded angles: theta1 = {:.2f} deg, theta2 = {:.2f} deg"
                .format(theta1_deg, theta2_deg)
                )

            x, y = forward_kinematics(theta1_deg, theta2_deg)
            print(
                    "\nCommanded position: x = {:.2f} cm, y = {:.2f} cm"
                    .format(x, y)
                )
            
            move_to_angles(theta1_deg, theta2_deg)

        elif choice == '2':
            measure_distance()

        elif choice == '3':
            measure_angle()

        elif choice == '4':
            move_to_position()
            
        elif choice == '5':
            midpoint()

        elif choice.lower() == 'c':
            calibrate_limits()

        elif choice.lower() == 'r':
            release_motors()

        elif choice.lower() == 'z':
            zero_arm()

        elif choice.lower() == 'cp':
            current_position()

        elif choice.lower() == 'q':
            print("\n================ EXIT ================\n")
            print("Exiting...")
            break

        else:
            print("\n================ INVALID INPUT ================\n")
            print("Invalid selection. Please choose a valid menu option.\n")


if __name__ == '__main__':
    main()