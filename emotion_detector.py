from deepface import DeepFace
import cv2

class EmotionDetector:
    def __init__(self):
        # Emotions that might indicate a student is struggling/confused
        self.struggle_emotions = ['sad', 'angry', 'fear', 'disgust']
        
    def get_struggle_index(self, frame):
        """
        Analyzes the frame and returns a boolean indicating struggle,
        along with the dominant emotion.
        """
        try:
            # We enforce enforce_detection=False so it doesn't crash if the face is briefly lost
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            dominant_emotion = result[0]['dominant_emotion']
            
            # If the emotion is in our struggle list, return True
            is_struggling = dominant_emotion in self.struggle_emotions
            return is_struggling, dominant_emotion
            
        except Exception as e:
            return False, "neutral"
