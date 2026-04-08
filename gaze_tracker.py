import mediapipe as mp
import cv2
import numpy as np

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Need this for irises
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_gaze_coordinates(self, frame, screen_w, screen_h):
        """
        Estimates where the user is looking on the screen.
        Returns (x, y) coordinates.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Use nose tip (landmark 1) as a simplified proxy for head pose/gaze focus
            # In a production app, you would calculate the vector between the iris and eyeball center.
            nose = landmarks[1]
            
            # Invert X because the webcam is mirrored
            x = int((1 - nose.x) * screen_w)
            # Map Y directly
            y = int(nose.y * screen_h)
            
            # Constrain to screen bounds
            x = max(0, min(x, screen_w - 1))
            y = max(0, min(y, screen_h - 1))
            
            return (x, y)
            
        return None
