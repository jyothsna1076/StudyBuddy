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

        self.struggle_emotions = ['angry', 'fear', 'disgust']

        self.eyes_closed_start_time = None
        self.missing_face_start_time = None
        self.is_sleeping = False

        self.emotion_history = deque(maxlen=10)
        self.state_history = deque(maxlen=10)

        # Calibration baselines
        self.center_yaw = 1.0
        self.center_pitch = 1.0
        self.ear_baseline = None
        self.mar_baseline = None
        
        # Personalized eyebrow baselines
        self.brow_ratio_baseline = None
        self.brow_height_left_baseline = None
        self.brow_height_right_baseline = None

        # Performance
        self.frame_counter = 0
        self.last_emotion = "neutral"

    # ---------------- CALIBRATION ---------------- #
    def calibrate_off_screen(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)

        if results.face_landmarks:
            nose = results.face_landmarks.landmark[1]
            left_face = results.face_landmarks.landmark[234]
            right_face = results.face_landmarks.landmark[454]
            top_head = results.face_landmarks.landmark[10]
            chin = results.face_landmarks.landmark[152]

            # Yaw
            dist_left = abs(nose.x - left_face.x)
            dist_right = abs(nose.x - right_face.x)
            self.center_yaw = dist_left / (dist_right if dist_right > 0 else 0.001)

            # Pitch
            dist_up = abs(nose.y - top_head.y)
            dist_down = abs(chin.y - nose.y)
            self.center_pitch = dist_up / (dist_down if dist_down > 0 else 0.001)

            # Baselines
            self.ear_baseline = self.calculate_ear(results.face_landmarks)
            self.mar_baseline = self.calculate_mar(results.face_landmarks)

            # CALIBRATE EYEBROWS
            inner_left_brow = results.face_landmarks.landmark[107]
            inner_right_brow = results.face_landmarks.landmark[336]
            left_eye_top = results.face_landmarks.landmark[159].y
            right_eye_top = results.face_landmarks.landmark[386].y
            
            face_width = abs(left_face.x - right_face.x) # Get face width
            brow_dist = abs(inner_left_brow.x - inner_right_brow.x)
            
            self.brow_ratio_baseline = brow_dist / (face_width if face_width > 0 else 0.001)
            
            # THE FIX: Normalize vertical height using face_width
            self.brow_height_left_baseline = abs(inner_left_brow.y - left_eye_top) / (face_width if face_width > 0 else 0.001)
            self.brow_height_right_baseline = abs(inner_right_brow.y - right_eye_top) / (face_width if face_width > 0 else 0.001)

            return True
        return False

    # ---------------- FEATURES ---------------- #
    def calculate_ear(self, face_landmarks):
        def dist(p1, p2): return math.dist([p1.x, p1.y], [p2.x, p2.y])

        left_v = dist(face_landmarks.landmark[159], face_landmarks.landmark[145])
        left_h = dist(face_landmarks.landmark[33], face_landmarks.landmark[133])

        right_v = dist(face_landmarks.landmark[386], face_landmarks.landmark[374])
        right_h = dist(face_landmarks.landmark[362], face_landmarks.landmark[263])

        ear_left = left_v / left_h if left_h > 0 else 0
        ear_right = right_v / right_h if right_h > 0 else 0

        return (ear_left + ear_right) / 2.0

    def calculate_mar(self, face_landmarks):
        def dist(p1, p2): return math.dist([p1.x, p1.y], [p2.x, p2.y])

        mouth_v = dist(face_landmarks.landmark[13], face_landmarks.landmark[14])
        mouth_h = dist(face_landmarks.landmark[78], face_landmarks.landmark[308])

        return mouth_v / mouth_h if mouth_h > 0 else 0

    def check_head_pose(self, face_landmarks):
        nose = face_landmarks.landmark[1]
        left_face = face_landmarks.landmark[234]
        right_face = face_landmarks.landmark[454]
        top_head = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]

        dist_left = abs(nose.x - left_face.x)
        dist_right = abs(nose.x - right_face.x)
        yaw_ratio = dist_left / (dist_right if dist_right > 0 else 0.001)

        dist_up = abs(nose.y - top_head.y)
        dist_down = abs(chin.y - nose.y)
        pitch_ratio = dist_up / (dist_down if dist_down > 0 else 0.001)

        looking_sideways = yaw_ratio > (self.center_yaw * 2.0) or yaw_ratio < (self.center_yaw * 0.5)
        looking_up = pitch_ratio < (self.center_pitch * 0.85)

        return looking_sideways or looking_up

    def check_body_language(self, pose_landmarks, face_landmarks=None):
        if not pose_landmarks:
            return False, False, False

        nose = pose_landmarks.landmark[0]
        l_shoulder = pose_landmarks.landmark[11]
        r_shoulder = pose_landmarks.landmark[12]
        
        # Pulling wrist + fingers to ensure hands on eyes are caught
        l_hand_points = [pose_landmarks.landmark[i] for i in [15, 17, 19, 21]]
        r_hand_points = [pose_landmarks.landmark[i] for i in [16, 18, 20, 22]]
        
        l_highest_y = min([p.y for p in l_hand_points])
        r_highest_y = min([p.y for p in r_hand_points])

        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        is_slouching = (avg_shoulder_y - nose.y) < 0.15

        left_hand_on_face = False
        right_hand_on_face = False

        if face_landmarks:
            top_head_y = face_landmarks.landmark[10].y
            chin_y = face_landmarks.landmark[152].y
            left_face_x = face_landmarks.landmark[234].x
            right_face_x = face_landmarks.landmark[454].x

            margin_x = 0.1
            # Original structure kept, but tracking the highest finger point instead of just the wrist
            # Also checking if it goes above the nose as requested
            if (l_highest_y < nose.y) or ((top_head_y - 0.1 < l_highest_y < chin_y) and (left_face_x - margin_x < pose_landmarks.landmark[15].x < right_face_x + margin_x)):
                left_hand_on_face = True
            if (r_highest_y < nose.y) or ((top_head_y - 0.1 < r_highest_y < chin_y) and (left_face_x - margin_x < pose_landmarks.landmark[16].x < right_face_x + margin_x)):
                right_hand_on_face = True
        else:
            if l_highest_y < nose.y: left_hand_on_face = True
            if r_highest_y < nose.y: right_hand_on_face = True

        both_hands = left_hand_on_face and right_hand_on_face
        one_hand = left_hand_on_face or right_hand_on_face

        return is_slouching, one_hand, both_hands

    # ---------------- MAIN LOGIC ---------------- #
    def get_struggle_index(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)

        self.frame_counter += 1

        eyes_closed = False
        face_missing = False
        is_yawning = False
        looking_away = False
        is_slouching = False
        one_hand = False
        both_hands = False
        
        # Micro-expressions
        is_confused_squeeze = False
        brows_furrowed = False

        # FACE
        if results.face_landmarks:
            self.missing_face_start_time = None
            
            # --- THE FIX 1: Actually call the looking away check! ---
            looking_away = self.check_head_pose(results.face_landmarks)

            ear = self.calculate_ear(results.face_landmarks)
            mar = self.calculate_mar(results.face_landmarks)

            # SLEEPY LOGIC (Relaxed baseline and reduced time to 1.0s)
            if ear < 0.22 or (self.ear_baseline and ear < self.ear_baseline * 0.80):
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = time.time()
                elif time.time() - self.eyes_closed_start_time > 1.0: # Reduced from 2.0
                    eyes_closed = True
            else:
                self.eyes_closed_start_time = None

            # YAWNING LOGIC
            if self.mar_baseline and (mar > self.mar_baseline * 1.5):
                if ear < 0.25: 
                    is_yawning = True

            # EYEBROW LOGIC 
            inner_left_brow = results.face_landmarks.landmark[107]
            inner_right_brow = results.face_landmarks.landmark[336]
            left_face = results.face_landmarks.landmark[234]
            right_face = results.face_landmarks.landmark[454]

            # Check if the face is hitting the edge of the camera frame
            is_clipping_edge = left_face.x < 0.05 or right_face.x > 0.95

            # Only run the eyebrow math if they are NOT at the edge of the screen
            if not is_clipping_edge:
                face_width = abs(left_face.x - right_face.x)
                brow_dist = abs(inner_left_brow.x - inner_right_brow.x)
                brow_ratio = brow_dist / (face_width if face_width > 0 else 0.001)

                left_eye_top = results.face_landmarks.landmark[159].y
                right_eye_top = results.face_landmarks.landmark[386].y
                
                brow_height_left = abs(inner_left_brow.y - left_eye_top) / (face_width if face_width > 0 else 0.001)
                brow_height_right = abs(inner_right_brow.y - right_eye_top) / (face_width if face_width > 0 else 0.001)

                if self.brow_ratio_baseline is not None:
                    # Confused (Vertical Lines)
                    if brow_ratio < self.brow_ratio_baseline * 0.94:
                        is_confused_squeeze = True
                        
                    # Squinting / Frowning
                    elif (brow_height_left < self.brow_height_left_baseline * 0.85) or \
                         (brow_height_right < self.brow_height_right_baseline * 0.85):
                        brows_furrowed = True
                        
        # --- THE FIX 2: Trigger face missing if no landmarks are found ---
        else:
            face_missing = True
            
        # BODY
        if results.pose_landmarks:
            is_slouching, one_hand, both_hands = self.check_body_language(
                results.pose_landmarks,
                results.face_landmarks if results.face_landmarks else None
            )

        # EMOTION
        if self.frame_counter % 10 == 0 and not face_missing and not looking_away:
            try:
                df_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                self.last_emotion = df_result[0]['dominant_emotion']
            except:
                pass

        self.emotion_history.append(self.last_emotion)

        weights = {'angry': 3, 'fear': 2, 'disgust': 2}
        emotion_score = sum(weights.get(e, 0) for e in self.emotion_history)

        # ---------------- STATUS ---------------- #
        if face_missing:
            status = "left screen"
        elif looking_away:
            status = "looking away"
        elif is_yawning:  
            status = "yawning"
        elif eyes_closed:
            status = "sleepy"
        elif both_hands:
            status = "highly stressed (hands on face/eyes)"
        elif one_hand:
            status = "stressed (hand on head/eyes)"
        elif is_confused_squeeze:
            status = "confused (vertical lines)"
        elif brows_furrowed:
            status = "stressed (squinting/frowning)"
        elif is_slouching:
            status = "slouching"
        elif emotion_score >= 5: 
            status = self.last_emotion
        else:
            status = "focused"

        # ---------------- FIXED NO-LAG LOGIC ---------------- #
        if status == "focused":
            struggle_level = "low"
        else:
            struggle_level = "high"

        return struggle_level, status