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

        self.emotion_history = deque(maxlen=10)
        self.state_history = deque(maxlen=10)

        # Calibration baselines
        self.center_yaw = 1.0
        self.center_pitch = 1.0
        self.ear_baseline = None
        self.mar_baseline = None

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
        l_wrist = pose_landmarks.landmark[15]
        r_wrist = pose_landmarks.landmark[16]

        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        is_slouching = (avg_shoulder_y - nose.y) < 0.15

        left_eye_y = None
        right_eye_y = None

        if face_landmarks:
            left_eye_y = face_landmarks.landmark[159].y
            right_eye_y = face_landmarks.landmark[386].y

        left_hand_eye = False
        right_hand_eye = False

        if left_eye_y is not None:
            left_hand_eye = abs(l_wrist.y - left_eye_y) < 0.05

        if right_eye_y is not None:
            right_hand_eye = abs(r_wrist.y - right_eye_y) < 0.05

        both_eyes = left_hand_eye and right_hand_eye
        one_eye = left_hand_eye or right_hand_eye

        return is_slouching, one_eye, both_eyes

    # ---------------- MAIN LOGIC ---------------- #
    def get_struggle_index(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)

        self.frame_counter += 1

        eyes_closed = False
        face_missing = False
        mouth_open = False
        looking_away = False
        is_slouching = False
        one_eye = False
        both_eyes = False

        # FACE
        if results.face_landmarks:
            self.missing_face_start_time = None

            ear = self.calculate_ear(results.face_landmarks)
            mar = self.calculate_mar(results.face_landmarks)

            if self.ear_baseline and ear < self.ear_baseline * 0.7:
                eyes_closed = True

            if self.mar_baseline and mar > self.mar_baseline * 1.5:
                mouth_open = True

            looking_away = self.check_head_pose(results.face_landmarks)

        else:
            if self.missing_face_start_time is None:
                self.missing_face_start_time = time.time()
            elif time.time() - self.missing_face_start_time > 0.5:
                face_missing = True

        # BODY
        if results.pose_landmarks:
            is_slouching, one_eye, both_eyes = self.check_body_language(
                results.pose_landmarks,
                results.face_landmarks if results.face_landmarks else None
            )

        # EMOTION (optimized)
        if self.frame_counter % 10 == 0 and not face_missing and not looking_away:
            try:
                df_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                self.last_emotion = df_result[0]['dominant_emotion']
            except:
                pass

        self.emotion_history.append(self.last_emotion)

        weights = {'sad': 2, 'angry': 3, 'fear': 2, 'disgust': 2}
        emotion_score = sum(weights.get(e, 0) for e in self.emotion_history)

        # ---------------- STATUS ---------------- #
        if face_missing:
            status = "left screen"
        elif looking_away:
            status = "looking away"
        elif eyes_closed:
            status = "sleepy"
        elif both_eyes:
            status = "high stress (covering eyes)"
        elif one_eye:
            status = "tired (hand on eye)"
        elif is_slouching:
            status = "slouching"
        elif emotion_score > 10:
            status = self.last_emotion
        else:
            status = "focused"

        # ---------------- YOUR RULE ---------------- #
        # anything not focused → HIGH
        self.state_history.append(status)

        non_focus_count = sum(1 for s in self.state_history if s != "focused")

        if non_focus_count >= 5:
            struggle_level = "high"
        else:
            struggle_level = "low"

        return struggle_level, status