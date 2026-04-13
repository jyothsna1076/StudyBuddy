"""
CalibrationManager Module
Orchestrates the initial setup phase for both the EmotionDetector and the GazeTracker.
Establishes user-specific biomechanical baselines to ensure tracking accuracy 
across different users, seating positions, and hardware configurations.
"""

import cv2
import numpy as np

class CalibrationManager:
    def __init__(self, gaze_tracker, emotion_detector):
        # References to the main tracking engines
        self.gaze_tracker = gaze_tracker
        self.emotion_detector = emotion_detector

    # ---------------- CAMERA CALIBRATION ---------------- #
    def calibrate_camera_center(self, cap, screen_w, screen_h):
        """
        Forces the user to look directly at the webcam to establish a neutral 'zero' state 
        for facial geometry (Pitch, Yaw, EAR, MAR).
        """
        print("Look at the WEBCAM and press SPACE")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Create a blank white canvas for the UI
            canvas = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255

            text = "Look at WEBCAM & press SPACE"
            cv2.putText(canvas, text, (screen_w//4, screen_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Calibration", canvas)

            # ASCII 32 is the SPACE bar. Waits for user confirmation.
            if cv2.waitKey(1) & 0xFF == 32:
                # Trigger the emotion detector's baseline calculation
                if self.emotion_detector.calibrate_off_screen(frame):
                    print("Camera calibration done")
                    break
                else:
                    # Failsafe if the face is occluded or out of frame
                    print("Face not detected, try again")

    # ---------------- GAZE CALIBRATION ---------------- #
    def _get_eye_relative_pos(self, landmarks):
        """
        Extracts the relative vector between the irises and the nose anchor.
        Duplicated from the GazeTracker to allow independent calibration logic.
        """
        iris_x = (landmarks[468].x + landmarks[473].x) / 2
        iris_y = (landmarks[468].y + landmarks[473].y) / 2

        anchor_x = landmarks[6].x
        anchor_y = landmarks[6].y

        return iris_x - anchor_x, iris_y - anchor_y

    def calibrate_gaze(self, cap, screen_w, screen_h):
        """
        Executes a standard 5-point calibration protocol (4 corners + center).
        Maps the physical limitations of the user's eye movement to the digital screen space.
        """
        # Force full screen to ensure the extreme corners of the monitor are targeted
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Define the 5 target points (5% margin from edges to prevent webcam occlusion)
        corners = [
            (int(screen_w * 0.05), int(screen_h * 0.05), "TOP_LEFT"),
            (int(screen_w * 0.95), int(screen_h * 0.05), "TOP_RIGHT"),
            (int(screen_w * 0.05), int(screen_h * 0.95), "BOTTOM_LEFT"),
            (int(screen_w * 0.95), int(screen_h * 0.95), "BOTTOM_RIGHT"),
            (int(screen_w * 0.5), int(screen_h * 0.5), "CENTER")
        ]

        results_map = {}

        for (sx, sy, name) in corners:
            samples_x, samples_y = [], []
            confirmed = False

            while not confirmed:
                ret, frame = cap.read()
                if not ret:
                    break

                # High-contrast black background to reduce ocular strain and isolate pupil movement
                calib_bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

                # Draw a crosshair target at the designated coordinates
                cv2.line(calib_bg, (sx-25, sy), (sx+25, sy), (255, 255, 255), 1)
                cv2.line(calib_bg, (sx, sy-25), (sx, sy+25), (255, 255, 255), 1)
                cv2.circle(calib_bg, (sx, sy), 10, (0, 0, 255), -1)

                cv2.putText(calib_bg, f"Look at {name} and press ENTER", (screen_w//2 - 250, screen_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("Calibration", calib_bg)

                # ASCII 13 is the ENTER key.
                if cv2.waitKey(1) & 0xFF == 13:
                    # --- STATISTICAL AVERAGING ---
                    # Capture 25 sequential frames while the user stares at the target.
                    # This mitigates the impact of micro-saccades (involuntary eye twitches) and camera noise.
                    for _ in range(25):
                        ret, frame = cap.read()
                        res = self.gaze_tracker.face_mesh.process(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        )

                        if res.multi_face_landmarks:
                            rel_x, rel_y = self._get_eye_relative_pos(
                                res.multi_face_landmarks[0].landmark
                            )
                            samples_x.append(rel_x)
                            samples_y.append(rel_y)

                    if samples_x:
                        # Store the statistical mean of the gathered samples
                        results_map[name] = (np.mean(samples_x), np.mean(samples_y))
                        confirmed = True

        # --- SPATIAL MAPPING INJECTION ---
        # Inject the calculated boundaries directly into the active GazeTracker instance.
        gt = self.gaze_tracker

        # Define the absolute center (0, 0 coordinate equivalent)
        gt.center_rel_x = results_map["CENTER"][0]
        gt.center_rel_y = results_map["CENTER"][1]

        # Calculate the maximum physical displacement vectors (deltas) from the center to the edges.
        # Taking the absolute difference prevents negative bounds.
        gt.range_x_left = abs(results_map["CENTER"][0] - results_map["TOP_LEFT"][0])
        gt.range_x_right = abs(results_map["TOP_RIGHT"][0] - results_map["CENTER"][0])
        gt.range_y_top = abs(results_map["CENTER"][1] - results_map["TOP_LEFT"][1])
        gt.range_y_bottom = abs(results_map["BOTTOM_LEFT"][1] - results_map["CENTER"][1])

        cv2.destroyWindow("Calibration")
        print("Gaze Calibration Complete")

    # ---------------- FULL PIPELINE ---------------- #
    def run_full_calibration(self, cap, screen_w, screen_h):
        """
        Executes the master sequence for system readiness.
        """
        self.calibrate_camera_center(cap, screen_w, screen_h)
        self.calibrate_gaze(cap, screen_w, screen_h)