import math
import time
from typing import Any, Dict, Optional
import numpy as np

from .driver import PCA9685
from .geometry import find_circle_intersections, calculate_angle, select_intersection_point


class SCARAController:
    def __init__(
        self,
        pwm: Any = PCA9685(0x40, debug=False),
        motor1_channel: int = 0,
        motor2_channel: int = 1,
        lift_servo_channel: int = 2,
        x: float = 0.0,
        y: float = 100.0,
        engine_spacing: float = 160.0,
        arm: float = 180.0,
        forearm: float = 260.0,
        max_angle: float = 270.0,
        min_pulse: int = 2590,
        max_pulse: int = 570,
        motor1_angle_offset: float = 68.0,
        motor2_angle_offset: float = 214.0,
    ) -> None:
        self.pwm: Any = pwm
        self.pwm.setPWMFreq(50)
        self.motor1_channel: int = motor1_channel
        self.motor2_channel: int = motor2_channel
        self.lift_servo_channel: int = lift_servo_channel
        self.x: float = x
        self.y: float = y
        self.engine_spacing: float = engine_spacing
        self.arm: float = arm
        self.forearm: float = forearm

        self.max_angle: float = max_angle
        self.min_pulse: int = min_pulse
        self.max_pulse: int = max_pulse
        self.motor1_angle_offset: float = motor1_angle_offset
        self.motor2_angle_offset: float = motor2_angle_offset

        self.x1: float = -engine_spacing / 2
        self.y1: float = -50.0
        self.x2: float = engine_spacing / 2
        self.y2: float = -50.0

    def angle_to_pulse(self, angle, motor):
        if motor == "motor1":
            angle_offset = self.motor1_angle_offset
        elif motor == "motor2":
            angle_offset = self.motor2_angle_offset
        else:
            raise ValueError("Invalid motor identifier. Use 'motor1' or 'motor2'.")

        adjusted_angle = angle - angle_offset
        adjusted_angle = adjusted_angle % 360

        if not (0 <= adjusted_angle <= self.max_angle):
            raise ValueError(f"Angle out of range for {motor}. Adjusted angle should be between 0 and {self.max_angle} degrees.")

        pulse = round(((adjusted_angle) / (self.max_angle)) * (self.max_pulse - self.min_pulse) + self.min_pulse)
        return pulse

    def flat_move(self, target_x: float, target_y: float) -> Optional[Dict[str, Any]]:
        """
        Executes a flat (planar) move of the SCARA robot arm to the specified (x, y) target.

        Parameters
        ----------
        target_x : float
            Target X position in mm.
        target_y : float
            Target Y position in mm.

        Returns
        -------
        dict | None
            Dictionary containing geometric data of the movement, or None if the move is impossible.
        """
        if target_y < 0:
            # Physical constraint
            return None

        r1 = r2 = self.arm

        # Compute intersections for each arm using numpy-based functions
        intersections1 = find_circle_intersections(
            np.array([self.x1, self.y1]), r1,
            np.array([target_x, target_y]), self.forearm
        )
        intersections2 = find_circle_intersections(
            np.array([self.x2, self.y2]), r2,
            np.array([target_x, target_y]), self.forearm
        )

        # Validate intersection results
        if intersections1 is None or intersections2 is None:
            return None

        selected_intersection1 = select_intersection_point(intersections1)
        selected_intersection2 = select_intersection_point(intersections2)

        if selected_intersection1 is None or selected_intersection2 is None:
            return None

        angle1 = calculate_angle(np.array([self.x1, self.y1]), selected_intersection1)
        angle2 = calculate_angle(np.array([self.x2, self.y2]), selected_intersection2)

        if angle1 is None or angle2 is None:
            return None

        # Check workspace constraints
        if angle1 < 45 or angle1 > 315:
            return None
        if not (angle2 <= 135 or angle2 >= 225):
            return None

        # Update TCP (tool center point) position
        self.x = target_x
        self.y = target_y

        # Send servo pulses
        pulse1 = self.angle_to_pulse(angle1, "motor1")
        self.pwm.setServoPulse(self.motor1_channel, pulse1)

        pulse2 = self.angle_to_pulse(angle2, "motor2")
        self.pwm.setServoPulse(self.motor2_channel, pulse2)

        # Return geometry for visualization / debug
        return {
            "x1": self.x1, "y1": self.y1, "r1": r1,
            "x2": self.x2, "y2": self.y2, "r2": r2,
            "x3": target_x, "y3": target_y, "r3": self.forearm,
            "intersection1": selected_intersection1,
            "intersection2": selected_intersection2,
            "angle1": angle1,
            "angle2": angle2,
        }

    def interpolated_flat_move(
        self,
        target_x: float,
        target_y: float,
        steps: int = 10,
        delay: float = 0.1
    ) -> bool:
        """
        Moves the SCARA robot in a straight-line interpolation (in XY plane)
        from the current (self.x, self.y) to (target_x, target_y),
        executing `steps` intermediate positions with a delay between them.

        Parameters
        ----------
        target_x : float
            Target X position in mm.
        target_y : float
            Target Y position in mm.
        steps : int, default=10
            Number of interpolation steps (including final position).
        delay : float, default=0.1
            Delay in seconds between consecutive steps.

        Returns
        -------
        bool
            True if the full interpolated path was executed,
            False if any intermediate move was not possible.
        """
        x_values = np.linspace(self.x, target_x, steps)
        y_values = np.linspace(self.y, target_y, steps)

        for x_val, y_val in zip(x_values, y_values):
            move_data = self.flat_move(float(x_val), float(y_val))
            if move_data is None:
                return False
            time.sleep(delay)

        return True

    def lift_robot(
        self,
        height_cm: float = 10.0,
        lift_time: float = 10.0,
        lift_pulse: int = 1500,
        stop_pulse: int = 1550
    ) -> bool:
        """
        Lifts the robot by activating the vertical servo for a specified duration.

        Parameters
        ----------
        height_cm : float
            Height in centimeters to lift (informational, not directly used in timing).
        lift_time : float
            Duration of lifting in seconds.
        lift_pulse : int
            PWM pulse to activate the lifting motion.
        stop_pulse : int
            PWM pulse to stop the servo after lifting.

        Returns
        -------
        bool
            True if the lift action was executed successfully.
        """
        try:
            self.pwm.setServoPulse(self.lift_servo_channel, lift_pulse)
            time.sleep(lift_time)
            self.pwm.setServoPulse(self.lift_servo_channel, stop_pulse)
            return True
        except Exception:
            return False


    def lower_robot(
        self,
        height_cm: float = 10.0,
        lower_time: float = 10.0,
        lower_pulse: int = 1610,
        stop_pulse: int = 1550
    ) -> bool:
        """
        Lowers the robot by activating the vertical servo for a specified duration.

        Parameters
        ----------
        height_cm : float
            Height in centimeters to lower (informational, not directly used in timing).
        lower_time : float
            Duration of lowering in seconds.
        lower_pulse : int
            PWM pulse to activate the lowering motion.
        stop_pulse : int
            PWM pulse to stop the servo after lowering.

        Returns
        -------
        bool
            True if the lowering action was executed successfully.
        """
        try:
            self.pwm.setServoPulse(self.lift_servo_channel, lower_pulse)
            time.sleep(lower_time)
            self.pwm.setServoPulse(self.lift_servo_channel, stop_pulse)
            return True
        except Exception:
            return False
