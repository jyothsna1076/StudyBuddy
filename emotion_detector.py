from deepface import DeepFace
import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
import math

class EmotionDetector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.struggle_emotions = ['sad', 'angry', 'fear', 'disgust']
        
        self.eyes_closed_start_time = None
        self.missing_face_start_time = None 
        self.is_sleeping = False
        self.emotion_history = deque(maxlen=5) 

        # NEW: Custom Calibration Baselines
        self.center_yaw = 1.0
        self.center_pitch = 1.0

    def calibrate_off_screen(self, frame):
        """Captures the user's head pose when looking perfectly at the camera."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        if results.face_landmarks:
            nose = results.face_landmarks.landmark[1]
            left_face = results.face_landmarks.landmark[234] 
            right_face = results.face_landmarks.landmark[454]
            top_head = results.face_landmarks.landmark[10]
            chin = results.face_landmarks.landmark[152]

            # Calculate Center Yaw (Left/Right balance)
            dist_left = abs(nose.x - left_face.x)
            dist_right = abs(nose.x - right_face.x)
            self.center_yaw = dist_left / (dist_right if dist_right > 0 else 0.001)

            # Calculate Center Pitch (Up/Down balance)
            dist_up = abs(nose.y - top_head.y)
            dist_down = abs(chin.y - nose.y)
            self.center_pitch = dist_up / (dist_down if dist_down > 0 else 0.001)
            return True
        return False

    def calculate_ear(self, face_landmarks):
        def dist(p1, p2): return math.dist([p1.x, p1.y], [p2.x, p2.y])
        left_v = dist(face_landmarks.landmark[159], face_landmarks.landmark[145])
        left_h = dist(face_landmarks.landmark[33], face_landmarks.landmark[133])
        ear_left = left_v / left_h if left_h > 0 else 0

        right_v = dist(face_landmarks.landmark[386], face_landmarks.landmark[374])
        right_h = dist(face_landmarks.landmark[362], face_landmarks.landmark[263])
        ear_right = right_v / right_h if right_h > 0 else 0

        return (ear_left + ear_right) / 2.0

    def calculate_mar(self, face_landmarks):
        def dist(p1, p2): return math.dist([p1.x, p1.y], [p2.x, p2.y])
        mouth_v = dist(face_landmarks.landmark[13], face_landmarks.landmark[14])
        mouth_h = dist(face_landmarks.landmark[78], face_landmarks.landmark[308])
        return mouth_v / mouth_h if mouth_h > 0 else 0

    def check_head_pose(self, face_landmarks):
        """Checks if the head deviated too far from the calibrated center point."""
        nose = face_landmarks.landmark[1]
        left_face = face_landmarks.landmark[234] 
        right_face = face_landmarks.landmark[454]
        top_head = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]

        # Current Yaw
        dist_left = abs(nose.x - left_face.x)
        dist_right = abs(nose.x - right_face.x)
        yaw_ratio = dist_left / (dist_right if dist_right > 0 else 0.001)

        # Current Pitch
        dist_up = abs(nose.y - top_head.y)
        dist_down = abs(chin.y - nose.y)
        pitch_ratio = dist_up / (dist_down if dist_down > 0 else 0.001)

        # If Yaw deviates significantly from the calibrated center (Turned left/right)
        looking_sideways = yaw_ratio > (self.center_yaw * 2.0) or yaw_ratio < (self.center_yaw * 0.5)

        # Pitch decreases when looking UP. 
        # If they look HIGHER than the physical webcam, flag it. 
        # Looking down at the screen is perfectly fine!
        looking_up = pitch_ratio < (self.center_pitch * 0.85) 

        return looking_sideways or looking_up

    def check_body_language(self, pose_landmarks):
        if not pose_landmarks: return False, False
        nose = pose_landmarks.landmark[0]
        l_shoulder = pose_landmarks.landmark[11]
        r_shoulder = pose_landmarks.landmark[12]
        l_wrist = pose_landmarks.landmark[15]
        r_wrist = pose_landmarks.landmark[16]

        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        head_to_shoulder_dist = avg_shoulder_y - nose.y
        is_slouching = head_to_shoulder_dist < 0.15 
        hands_on_face = (l_wrist.y < nose.y) or (r_wrist.y < nose.y)

        return is_slouching, hands_on_face

    def get_struggle_index(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        eyes_closed, is_slouching, hands_on_face = False, False, False
        face_missing, mouth_open, looking_away = False, False, False
        
        if results.face_landmarks:
            self.missing_face_start_time = None 
            
            ear = self.calculate_ear(results.face_landmarks)
            if ear < 0.20:
                eyes_closed = True
                if self.eyes_closed_start_time is None: self.eyes_closed_start_time = time.time()
                elif time.time() - self.eyes_closed_start_time > 2.0: self.is_sleeping = True
            else:
                self.eyes_closed_start_time = None
                self.is_sleeping = False
                
            mar = self.calculate_mar(results.face_landmarks)
            if mar > 0.40: mouth_open = True

            looking_away = self.check_head_pose(results.face_landmarks)
        else:
            if self.missing_face_start_time is None: self.missing_face_start_time = time.time()
            elif time.time() - self.missing_face_start_time > 0.5:
                face_missing = True
                self.is_sleeping = False 
                self.eyes_closed_start_time = None

        if results.pose_landmarks:
            is_slouching, hands_on_face = self.check_body_language(results.pose_landmarks)

        dominant_emotion = "neutral"
        if not face_missing and not looking_away: 
            try:
                df_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                dominant_emotion = df_result[0]['dominant_emotion']
            except Exception:
                pass
            self.emotion_history.append(dominant_emotion)
            
        struggle_emotion_count = sum(1 for e in self.emotion_history if e in self.struggle_emotions)
        consistent_negative_emotion = struggle_emotion_count >= 3 

        struggle_level = "low"
        status_message = dominant_emotion

        if face_missing: struggle_level, status_message = "high", "distracted (left the screen)"
        elif looking_away: struggle_level, status_message = "high", "distracted (looking off-screen)"
        elif self.is_sleeping: struggle_level, status_message = "high", "sleepy (eyes closed)"
        elif hands_on_face: struggle_level, status_message = "high", "stress (hands on eyes/face)"
        elif mouth_open: struggle_level, status_message = "high", "yawning / showing teeth"
        elif is_slouching: struggle_level, status_message = "high", "slouching"
        elif consistent_negative_emotion:
            struggle_level = "high"
            if dominant_emotion == 'angry': status_message = "narrowed eyebrows / flinching"
            elif dominant_emotion == 'fear': status_message = "flinching / startled"
            else: status_message = dominant_emotion
        else:
            status_message = "focused / clear"

        return struggle_level, status_message