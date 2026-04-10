import mediapipe as mp
import cv2
import numpy as np
import time
class GazeTracker:
    def __init__(self):
        self.calib_left = 0.0
        self.calib_right = 1.0
        self.calib_top = 0.0
        self.calib_bottom = 1.0
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Need this for irises
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
  
    def calibrate(self, cap, screen_w, screen_h):
        # Corners to calibrate: (Screen X, Screen Y, Label)
        corners = [
            (int(screen_w * 0.1), int(screen_h * 0.1), "TOP_LEFT"),
            (int(screen_w * 0.9), int(screen_h * 0.1), "TOP_RIGHT"),
            (int(screen_w * 0.1), int(screen_h * 0.9), "BOTTOM_LEFT"),
            (int(screen_w * 0.9), int(screen_h * 0.9), "BOTTOM_RIGHT")
        ]
        
        calib_data = {"x": [], "y": []}
        results_map = {}

        for (sx, sy, name) in corners:
            print(f"Look at the dot at {name}...")
            samples_x = []
            samples_y = []
            
            # Give user time to move eyes to the dot
            start_time = time.time()
            while len(samples_x) < 20: # Collect 20 stable frames
                ret, frame = cap.read()
                if not ret: break
                
                # 1. Draw the calibration dot on the frame (or a UI window)
                cv2.circle(frame, (sx // 2, sy // 2), 10, (0, 0, 255), -1) # Adjust scale if frame != screen
                cv2.putText(frame, f"Stare at the RED DOT: {name}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Calibration", frame)
                cv2.waitKey(1)

                # 2. Extract iris data (Raw normalized values)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.face_mesh.process(rgb_frame)
                
                if res.multi_face_landmarks:
                    landmarks = res.multi_face_landmarks[0].landmark
                    # Only start collecting after 1 second to let eyes settle
                    if time.time() - start_time > 1.0:
                        gx = (landmarks[468].x + landmarks[473].x) / 2
                        gy = (landmarks[468].y + landmarks[473].y) / 2
                        samples_x.append(gx)
                        samples_y.append(gy)
            
            # Average the samples for this corner
            results_map[name] = (np.mean(samples_x), np.mean(samples_y))

        # Define the boundaries based on averaged samples
        self.calib_left = (results_map["TOP_LEFT"][0] + results_map["BOTTOM_LEFT"][0]) / 2
        self.calib_right = (results_map["TOP_RIGHT"][0] + results_map["BOTTOM_RIGHT"][0]) / 2
        self.calib_top = (results_map["TOP_LEFT"][1] + results_map["TOP_RIGHT"][1]) / 2
        self.calib_bottom = (results_map["BOTTOM_LEFT"][1] + results_map["BOTTOM_RIGHT"][1]) / 2
        
        cv2.destroyWindow("Calibration")
        print("Calibration Complete!")

    def get_gaze_coordinates(self, frame, screen_w, screen_h):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Center point of the irises in normalized camera space (0.0 to 1.0)
            gaze_x = (landmarks[468].x + landmarks[473].x) / 2
            gaze_y = (landmarks[468].y + landmarks[473].y) / 2

            # --- CALIBRATION MAPPING ---
            # Instead of multiplying by screen_w directly, we map the iris 
            # position relative to your recorded calibration boundaries.
            
            # Formula: (current - min) / (max - min)
            # This turns your eye movement into a clean 0.0 to 1.0 percentage of your screen.
            
            denom_x = (self.calib_right - self.calib_left)
            denom_y = (self.calib_bottom - self.calib_top)
            
            if denom_x == 0 or denom_y == 0:
                return None # Prevent division by zero if not calibrated
                
            relative_x = (gaze_x - self.calib_left) / denom_x
            relative_y = (gaze_y - self.calib_top) / denom_y

            # Invert X for mirrored webcam, then scale to pixels
            x = int((1 - relative_x) * screen_w)
            y = int(relative_y * screen_h)
            
            # Constrain to screen bounds
            x = max(0, min(x, screen_w - 1))
            y = max(0, min(y, screen_h - 1))
            
            return (x, y)
        return None