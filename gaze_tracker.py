import mediapipe as mp
import cv2
import numpy as np


class GazeTracker:
    def __init__(self):
        # Smoothing
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 0.2  # adjustable

        # Calibration values (will be set externally)
        self.center_rel_x = 0.0
        self.center_rel_y = 0.0

        self.range_x_left = 1e-6
        self.range_x_right = 1e-6
        self.range_y_top = 1e-6
        self.range_y_bottom = 1e-6

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    # ---------------- CORE FEATURE ---------------- #
    def _get_eye_relative_pos(self, landmarks):
        iris_x = (landmarks[468].x + landmarks[473].x) / 2
        iris_y = (landmarks[468].y + landmarks[473].y) / 2

        anchor_x = landmarks[6].x
        anchor_y = landmarks[6].y

        return iris_x - anchor_x, iris_y - anchor_y

    # ---------------- MAIN FUNCTION ---------------- #
    def get_gaze_coordinates(self, frame, screen_w, screen_h):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            curr_x, curr_y = self._get_eye_relative_pos(landmarks)

            # Offset from center
            diff_x = curr_x - self.center_rel_x
            diff_y = curr_y - self.center_rel_y

            # Prevent division errors
            rx_l = max(self.range_x_left, 1e-6)
            rx_r = max(self.range_x_right, 1e-6)
            ry_t = max(self.range_y_top, 1e-6)
            ry_b = max(self.range_y_bottom, 1e-6)

            # Quadrant scaling
            if diff_x < 0:
                norm_x = 0.5 + (diff_x / (rx_l * 2))
            else:
                norm_x = 0.5 + (diff_x / (rx_r * 2))

            if diff_y < 0:
                norm_y = 0.5 + (diff_y / (ry_t * 2))
            else:
                norm_y = 0.5 + (diff_y / (ry_b * 2))

            # Clamp normalized values
            norm_x = max(0, min(1, norm_x))
            norm_y = max(0, min(1, norm_y))

            # Convert to screen coords
            target_x = int((1 - norm_x) * screen_w)
            target_y = int(norm_y * screen_h)

            # Apply smoothing
            alpha = self.smoothing
            self.prev_x = int(self.prev_x * (1 - alpha) + target_x * alpha)
            self.prev_y = int(self.prev_y * (1 - alpha) + target_y * alpha)

            return (
                max(0, min(self.prev_x, screen_w - 1)),
                max(0, min(self.prev_y, screen_h - 1))
            )

        return None