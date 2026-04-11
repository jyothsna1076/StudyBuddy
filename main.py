import cv2
import numpy as np
from emotion_detector import EmotionDetector
from heatmap_generator import HeatmapGenerator
from mouse_tracker import MouseTracker 
from calibration import CalibrationManager

def draw_ui(canvas, state, level, color):
    cv2.putText(canvas, f"State: {str(state).upper()}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(canvas, f"Struggle: {str(level).upper()}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def main():
    # 1. Load Material
    study_material = cv2.imread('assets/study_material.png') 
    if study_material is None:
        print("Missing study material in assets/")
        return

    h, w = study_material.shape[:2]
    WIN_MATERIAL = "StudyBuddy"
    WIN_WEBCAM = "Webcam"

    # 2. Initialize Modules
    cap = cv2.VideoCapture(0)
    emotion_detector = EmotionDetector()
    heatmap = HeatmapGenerator(w, h)
    mouse = MouseTracker(WIN_MATERIAL)

    # 3. Calibration
    calibrator = CalibrationManager(None, emotion_detector) 
    calibrator.calibrate_camera_center(cap, w, h)
    cv2.destroyAllWindows() 
    cv2.waitKey(1)
    print("System Ready. Mouse tracking active. Press Q or click [X] to quit.")

    frame_count = 0
    struggle_level = "low"
    state = "neutral"
    color = (0, 255, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            result = emotion_detector.get_struggle_index(frame)
            struggle_level = result[0]
            state = result[1]
        
        frame_count += 1

        x, y = mouse.get_position()
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        canvas = study_material.copy()

        if str(struggle_level).lower() == "high":
            heatmap.add_struggle_point(x, y)
            color = (0, 0, 255) 
        else:
            color = (0, 255, 0) 

        cv2.circle(canvas, (x, y), 15, color, -1)

        overlay = heatmap.get_heatmap_overlay()
        if overlay is not None:
            gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            mask = gray > 0
            if mask.any():
                canvas[mask] = cv2.addWeighted(canvas[mask], 0.5, overlay[mask], 0.5, 0)

        # UI and Display
        draw_ui(canvas, state, struggle_level, color)
        cv2.imshow(WIN_WEBCAM, cv2.flip(frame, 1))
        cv2.imshow(WIN_MATERIAL, canvas)

        # --- EXIT LOGIC ---
        key = cv2.waitKey(1) & 0xFF
        
        # 1. Exit if 'q' is pressed
        if key == ord('q'):
            break
            
        # 2. Exit if the [X] button is clicked on either window
        if cv2.getWindowProperty(WIN_MATERIAL, cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty(WIN_WEBCAM, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()