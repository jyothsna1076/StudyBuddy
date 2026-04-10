import cv2
import numpy as np
import time
from gaze_tracker import GazeTracker
from emotion_detector import EmotionDetector
from heatmap_generator import HeatmapGenerator

def main():
    # 1. Load the study material (Ensure you have an image in the assets folder!)
    study_material = cv2.imread('assets/study_material.jpg')
    if study_material is None:
        print("Error: Could not load assets/study_material.jpg. Please add an image.")
        return
        
    study_h, study_w = study_material.shape[:2]

    # 2. Initialize AI Modules
    cap = cv2.VideoCapture(0)
    gaze_tracker = GazeTracker()
    emotion_detector = EmotionDetector()
    heatmap_gen = HeatmapGenerator(study_w, study_h)
    print("StudyBuddy initialized. Press 'q' to quit.")
    print("Starting Calibration. Please keep your head still.")
    gaze_tracker.calibrate(cap, study_w, study_h)
    
    # 3. MAIN TRACKING LOOP
    print("Calibration finished. Starting tracking...")
    
    
    frame_count = 0
    is_struggling = False
    current_emotion = "neutral"

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Display the webcam feed in a small window
        cv2.imshow('Webcam (You)', cv2.flip(frame, 1))

        # We only run Emotion Detection every 10 frames to prevent severe lag
        if frame_count % 10 == 0:
            is_struggling, current_emotion = emotion_detector.get_struggle_index(frame)
        
        frame_count += 1

        # Track Gaze relative to the dimensions of the study material
        gaze_coords = gaze_tracker.get_gaze_coordinates(frame, study_w, study_h)

        # Output canvas starts as a fresh copy of the study material
        display_canvas = study_material.copy()

        if gaze_coords:
            g_x, g_y = gaze_coords
            
            # If the user is struggling, add to the heatmap
            if is_struggling:
                heatmap_gen.add_struggle_point(g_x, g_y)
                
            # Draw a circle showing where the system THINKS you are looking
            color = (0, 0, 255) if is_struggling else (0, 255, 0) # Red if struggling, Green if okay
            cv2.circle(display_canvas, (g_x, g_y), 15, color, -1)

        # Generate and overlay the heatmap
        heatmap_layer = heatmap_gen.get_heatmap_overlay()
        
        # Blend the heatmap over the study material where heatmap data exists
        gray_heatmap = cv2.cvtColor(heatmap_layer, cv2.COLOR_BGR2GRAY)
        mask = gray_heatmap > 0
        if mask.any():
            display_canvas[mask] = cv2.addWeighted(display_canvas[mask], 0.5, heatmap_layer[mask], 0.5, 0)

        # Display HUD info
        cv2.putText(display_canvas, f"Emotion: {current_emotion}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_canvas, f"Struggle Index: {'HIGH' if is_struggling else 'LOW'}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_struggling else (0, 255, 0), 2)

        # Show the main Study Material interface
        cv2.imshow('StudyBuddy - Material View', display_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
