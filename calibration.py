import cv2
import numpy as np

class CalibrationManager:
    def __init__(self, gaze_tracker, emotion_detector):
        self.gaze_tracker = gaze_tracker
        self.emotion_detector = emotion_detector

    # ---------------- CAMERA CALIBRATION ---------------- #
    def calibrate_camera_center(self, cap, screen_w, screen_h):
        print("Look at the WEBCAM and press SPACE")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            canvas = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255

            text = "Look at WEBCAM & press SPACE"
            cv2.putText(canvas, text, (screen_w//4, screen_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.imshow("Calibration", canvas)

            if cv2.waitKey(1) & 0xFF == 32:
                if self.emotion_detector.calibrate_off_screen(frame):
                    print("Camera calibration done")
                    break
                else:
                    print("Face not detected, try again")

    # ---------------- GAZE CALIBRATION ---------------- #
    def _get_eye_relative_pos(self, landmarks):
        iris_x = (landmarks[468].x + landmarks[473].x) / 2
        iris_y = (landmarks[468].y + landmarks[473].y) / 2

        anchor_x = landmarks[6].x
        anchor_y = landmarks[6].y

        return iris_x - anchor_x, iris_y - anchor_y

    def calibrate_gaze(self, cap, screen_w, screen_h):
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

                calib_bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

                cv2.line(calib_bg, (sx-25, sy), (sx+25, sy), (255,255,255), 1)
                cv2.line(calib_bg, (sx, sy-25), (sx, sy+25), (255,255,255), 1)
                cv2.circle(calib_bg, (sx, sy), 10, (0,0,255), -1)

                cv2.putText(calib_bg, f"Look at {name}", (screen_w//2 - 150, screen_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                cv2.imshow("Calibration", calib_bg)

                if cv2.waitKey(1) & 0xFF == 13:
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
                        results_map[name] = (np.mean(samples_x), np.mean(samples_y))
                        confirmed = True

        # Mapping
        self.gaze_tracker.calib_left = (results_map["TOP_LEFT"][0] + results_map["BOTTOM_LEFT"][0]) / 2
        self.gaze_tracker.calib_right = (results_map["TOP_RIGHT"][0] + results_map["BOTTOM_RIGHT"][0]) / 2
        self.gaze_tracker.calib_top = (results_map["TOP_LEFT"][1] + results_map["TOP_RIGHT"][1]) / 2
        self.gaze_tracker.calib_bottom = (results_map["BOTTOM_LEFT"][1] + results_map["BOTTOM_RIGHT"][1]) / 2

        self.gaze_tracker.center_rel_x = results_map["CENTER"][0]
        self.gaze_tracker.center_rel_y = results_map["CENTER"][1]
        gt = self.gaze_tracker

        gt.center_rel_x = results_map["CENTER"][0]
        gt.center_rel_y = results_map["CENTER"][1]

        gt.range_x_left = abs(results_map["CENTER"][0] - results_map["TOP_LEFT"][0])
        gt.range_x_right = abs(results_map["TOP_RIGHT"][0] - results_map["CENTER"][0])
        gt.range_y_top = abs(results_map["CENTER"][1] - results_map["TOP_LEFT"][1])
        gt.range_y_bottom = abs(results_map["BOTTOM_LEFT"][1] - results_map["CENTER"][1])

        cv2.destroyWindow("Calibration")
        print("Gaze Calibration Complete!")

    # ---------------- FULL PIPELINE ---------------- #
    def run_full_calibration(self, cap, screen_w, screen_h):
        self.calibrate_camera_center(cap, screen_w, screen_h)
        self.calibrate_gaze(cap, screen_w, screen_h)