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
        
        # 'angry' captures narrowed eyebrows/frustration. 'fear' captures flinching.
        self.struggle_emotions = ['sad', 'angry', 'fear', 'disgust']
        
        self.eyes_closed_start_time = None
        self.missing_face_start_time = None 
        self.is_sleeping = False
        self.emotion_history = deque(maxlen=5) 

    def calculate_ear(self, face_landmarks):
        """Calculates Eye Aspect Ratio (EAR) to detect closed eyes."""
        def dist(p1, p2):
            return math.dist([p1.x, p1.y], [p2.x, p2.y])

        left_v = dist(face_landmarks.landmark[159], face_landmarks.landmark[145])
        left_h = dist(face_landmarks.landmark[33], face_landmarks.landmark[133])
        ear_left = left_v / left_h if left_h > 0 else 0

        right_v = dist(face_landmarks.landmark[386], face_landmarks.landmark[374])
        right_h = dist(face_landmarks.landmark[362], face_landmarks.landmark[263])
        ear_right = right_v / right_h if right_h > 0 else 0

        return (ear_left + ear_right) / 2.0

    def calculate_mar(self, face_landmarks):
        """Calculates Mouth Aspect Ratio (MAR) to detect yawning or showing teeth."""
        def dist(p1, p2):
            return math.dist([p1.x, p1.y], [p2.x, p2.y])
            
        # Vertical distance: Inner top lip (13) to Inner bottom lip (14)
        mouth_v = dist(face_landmarks.landmark[13], face_landmarks.landmark[14])
        # Horizontal distance: Left corner of mouth (78) to Right corner (308)
        mouth_h = dist(face_landmarks.landmark[78], face_landmarks.landmark[308])
        
        return mouth_v / mouth_h if mouth_h > 0 else 0

    def check_body_language(self, pose_landmarks):
        """Checks for slouching or hands on head."""
        if not pose_landmarks:
            return False, False

        nose = pose_landmarks.landmark[0]
        l_shoulder = pose_landmarks.landmark[11]
        r_shoulder = pose_landmarks.landmark[12]
        l_wrist = pose_landmarks.landmark[15]
        r_wrist = pose_landmarks.landmark[16]

        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        head_to_shoulder_dist = avg_shoulder_y - nose.y
        is_slouching = head_to_shoulder_dist < 0.15 

        hands_on_head = (l_wrist.y < nose.y) or (r_wrist.y < nose.y)

        return is_slouching, hands_on_head

    def get_struggle_index(self, frame):
        """
        Analyzes the frame and returns (struggle_level, status_message).
        struggle_level is strictly 'low' or 'high'.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        eyes_closed = False
        is_slouching = False
        hands_on_head = False
        face_missing = False
        mouth_open = False
        
        if results.face_landmarks:
            self.missing_face_start_time = None 
            
            # Check Eyes
            ear = self.calculate_ear(results.face_landmarks)
            if ear < 0.20:
                eyes_closed = True
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = time.time()
                elif time.time() - self.eyes_closed_start_time > 2.0:
                    self.is_sleeping = True
            else:
                self.eyes_closed_start_time = None
                self.is_sleeping = False
                
            # Check Mouth (Yawning / Grimacing / Showing Teeth)
            mar = self.calculate_mar(results.face_landmarks)
            if mar > 0.40: # Threshold for a wide open mouth
                mouth_open = True
        else:
            if self.missing_face_start_time is None:
                self.missing_face_start_time = time.time()
            elif time.time() - self.missing_face_start_time > 0.5:
                face_missing = True
                self.is_sleeping = False 
                self.eyes_closed_start_time = None

        if results.pose_landmarks:
            is_slouching, hands_on_head = self.check_body_language(results.pose_landmarks)

        dominant_emotion = "neutral"
        if not face_missing: 
            try:
                df_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                dominant_emotion = df_result[0]['dominant_emotion']
            except Exception:
                pass
            self.emotion_history.append(dominant_emotion)
            
        struggle_emotion_count = sum(1 for e in self.emotion_history if e in self.struggle_emotions)
        consistent_negative_emotion = struggle_emotion_count >= 3 

        # --- BINARY HIGH/LOW HIERARCHY ---
        struggle_level = "low"
        status_message = dominant_emotion

        if face_missing:
            struggle_level = "high"
            status_message = "looking away / not visible"
        elif self.is_sleeping:
            struggle_level = "high"
            status_message = "sleeping / eyes closed > 2s"
        elif mouth_open:
            struggle_level = "high"
            status_message = "yawning / showing teeth"
        elif hands_on_head:
            struggle_level = "high"
            status_message = "frustrated (hands on head)"
        elif is_slouching:
            struggle_level = "high"
            status_message = "slouching"
        elif consistent_negative_emotion:
            struggle_level = "high"
            if dominant_emotion == 'angry':
                status_message = "narrowed eyebrows / flinching"
            elif dominant_emotion == 'fear':
                status_message = "flinching / startled"
            else:
                status_message = dominant_emotion
        else:
            # If nothing bad is detected, default to low.
            status_message = "focused / clear"

        return struggle_level, status_message