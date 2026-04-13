"""
GazeTracker Module
Uses Google's MediaPipe FaceMesh to track iris movements and map them 
to estimated screen coordinates.
"""

import mediapipe as mp
import cv2
import numpy as np


class GazeTracker:
    def __init__(self):
        # --- Smoothing Parameters ---
        self.prev_x, self.prev_y = 0, 0
        # Alpha value for Exponential Moving Average (EMA) smoothing.
        # Lower value = smoother but more lag; Higher value = more responsive but jittery.
        self.smoothing = 0.2  

        # --- Calibration Values ---
        # These are dynamically set during the initial setup phase.
        # Represents the user's eye position when looking exactly at the center of the screen.
        self.center_rel_x = 0.0
        self.center_rel_y = 0.0

        # Max movement ranges (Left, Right, Top, Bottom) relative to the center.
        # Initialized to a very small number (1e-6) to prevent ZeroDivisionError before calibration.
        self.range_x_left = 1e-6
        self.range_x_right = 1e-6
        self.range_y_top = 1e-6
        self.range_y_bottom = 1e-6

        # --- MediaPipe Initialization ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # CRITICAL: Must be True to output the Iris landmarks (468-477)
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    # ---------------- CORE FEATURE ---------------- #
    def _get_eye_relative_pos(self, landmarks):
        """
        Calculates the relative position of the eyes compared to the center of the face.
        This isolates eye movement from general head movement.
        """
        # Landmark 468 = Center of Left Iris
        # Landmark 473 = Center of Right Iris
        iris_x = (landmarks[468].x + landmarks[473].x) / 2
        iris_y = (landmarks[468].y + landmarks[473].y) / 2

        # Landmark 6 = Nose tip / Bridge of the nose (Serves as a stable anchor point)
        anchor_x = landmarks[6].x
        anchor_y = landmarks[6].y

        # Return the vector/distance between the irises and the nose anchor
        return iris_x - anchor_x, iris_y - anchor_y

    # ---------------- MAIN FUNCTION ---------------- #
    def get_gaze_coordinates(self, frame, screen_w, screen_h):
        """
        Processes a video frame, extracts iris positions, normalizes them against
        calibration bounds, and translates them into physical screen coordinates (X, Y).
        """
        # MediaPipe requires RGB images, OpenCV captures in BGR format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Get current relative eye position
            curr_x, curr_y = self._get_eye_relative_pos(landmarks)

            # Calculate how far the eyes have moved from the calibrated center
            diff_x = curr_x - self.center_rel_x
            diff_y = curr_y - self.center_rel_y

            # Prevent division errors (clamp minimum values)
            rx_l = max(self.range_x_left, 1e-6)
            rx_r = max(self.range_x_right, 1e-6)
            ry_t = max(self.range_y_top, 1e-6)
            ry_b = max(self.range_y_bottom, 1e-6)

            # --- Quadrant Scaling (Normalization) ---
            # We scale left/right and up/down separately because eye movement 
            # is not perfectly symmetrical. Normalizes values to a [0.0 - 1.0] scale.
            
            # X-Axis Normalization
            if diff_x < 0:
                norm_x = 0.5 + (diff_x / (rx_l * 2))
            else:
                norm_x = 0.5 + (diff_x / (rx_r * 2))

            # Y-Axis Normalization
            if diff_y < 0:
                norm_y = 0.5 + (diff_y / (ry_t * 2))
            else:
                norm_y = 0.5 + (diff_y / (ry_b * 2))

            # Clamp normalized values strictly between 0 and 1 to prevent cursor flying off-screen
            norm_x = max(0, min(1, norm_x))
            norm_y = max(0, min(1, norm_y))

            # --- Map to Screen Space ---
            # target_x uses (1 - norm_x) to mirror the X axis (webcams are mirrored by default)
            target_x = int((1 - norm_x) * screen_w)
            target_y = int(norm_y * screen_h)

            # --- Apply Exponential Moving Average (EMA) Smoothing ---
            # Prevents cursor/gaze jumping due to micro-movements or minor webcam noise
            alpha = self.smoothing
            self.prev_x = int(self.prev_x * (1 - alpha) + target_x * alpha)
            self.prev_y = int(self.prev_y * (1 - alpha) + target_y * alpha)

            # Return final coordinates clamped to actual screen dimensions
            return (
                max(0, min(self.prev_x, screen_w - 1)),
                max(0, min(self.prev_y, screen_h - 1))
            )

        # Return None if no face is detected in the frame
        return None