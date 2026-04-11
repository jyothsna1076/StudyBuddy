import cv2
import numpy as np
import time
from gaze_tracker import GazeTracker
from emotion_detector import EmotionDetector
from heatmap_generator import HeatmapGenerator

def run_calibration(cap, gaze_tracker, emotion_detector, canvas_w, canvas_h):
    print("Starting Calibration...")
    
    # --- STEP 1: OFF-SCREEN / WEBCAM CALIBRATION ---
    print("Calibration: Look directly at the WEBCAM LENS and press 'SPACE'.")
    while True:
        success, frame = cap.read()
        if not success: break
        
        calib_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255 
        instruction_text = "Look at the WEBCAM LENS (Not the screen!) & press SPACE"
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        cv2.putText(calib_canvas, instruction_text, ((canvas_w - text_size[0]) // 2, canvas_h // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('StudyBuddy - Material View', calib_canvas)
        
        if cv2.waitKey(1) & 0xFF == 32: 
            if emotion_detector.calibrate_off_screen(frame):
                print("Webcam Center captured!")
                # Flash Green
                cv2.rectangle(calib_canvas, (0,0), (canvas_w, canvas_h), (0, 255, 0), -1)
                cv2.imshow('StudyBuddy - Material View', calib_canvas)
                cv2.waitKey(500)
                break
            else:
                print("Couldn't see your face! Try again.")

    # --- STEP 2: 4-CORNER GAZE CALIBRATION ---
    margin = 40
    corners = [
        ("TOP-LEFT", (margin, margin)),
        ("TOP-RIGHT", (canvas_w - margin, margin)),
        ("BOTTOM-LEFT", (margin, canvas_h - margin)),
        ("BOTTOM-RIGHT", (canvas_w - margin, canvas_h - margin))
    ]
    
    recorded_ratios_x = []
    recorded_ratios_y = []

    for corner_name, point in corners:
        print(f"Calibration: Look at {corner_name} and press 'SPACE'.")
        while True:
            success, frame = cap.read()
            if not success: break
            
            calib_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255 
            instruction_text = f"Look at the RED DOT ({corner_name}) and press SPACE"
            text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            cv2.putText(calib_canvas, instruction_text, ((canvas_w - text_size[0]) // 2, canvas_h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.circle(calib_canvas, point, 20, (0, 0, 255), -1)
            cv2.imshow('StudyBuddy - Material View', calib_canvas)
            
            if cv2.waitKey(1) & 0xFF == 32: 
                ratios = gaze_tracker.get_raw_ratios(frame)
                if ratios:
                    recorded_ratios_x.append(ratios[0])
                    recorded_ratios_y.append(ratios[1])
                    print(f"{corner_name} captured!")
                    cv2.circle(calib_canvas, point, 20, (0, 255, 0), -1)
                    cv2.imshow('StudyBuddy - Material View', calib_canvas)
                    cv2.waitKey(300) 
                    break
                else:
                    print("Couldn't see your eyes! Try again.")

    if len(recorded_ratios_x) == 4:
        min_x, max_x = min(recorded_ratios_x), max(recorded_ratios_x)
        min_y, max_y = min(recorded_ratios_y), max(recorded_ratios_y)
        gaze_tracker.set_calibration(min_x, max_x, min_y, max_y)
        print(f"Calibration Complete! Min/Max X: {min_x:.3f}/{max_x:.3f}, Y: {min_y:.3f}/{max_y:.3f}")
    
    loading_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255 
    cv2.putText(loading_canvas, "Calibration Complete! Loading Study Material...", 
                (50, canvas_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    cv2.imshow('StudyBuddy - Material View', loading_canvas)
    cv2.waitKey(2000)

def main():
    study_material = cv2.imread('assets/study_material.png')
    if study_material is None:
        print("Error: Could not load assets/study_material.png. Please add an image.")
        return
        
    study_h, study_w = study_material.shape[:2]

    cap = cv2.VideoCapture(0)
    gaze_tracker = GazeTracker()
    emotion_detector = EmotionDetector()
    heatmap_gen = HeatmapGenerator(study_w, study_h)

    run_calibration(cap, gaze_tracker, emotion_detector, study_w, study_h)
    print("StudyBuddy initialized. Press 'q' to quit.")
    
    frame_count = 0
    struggle_level = "low"
    current_state = "focused / clear"

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        cv2.imshow('Webcam (You)', cv2.flip(frame, 1))

        if frame_count % 10 == 0:
            struggle_level, current_state = emotion_detector.get_struggle_index(frame)
        
        frame_count += 1
        gaze_coords = gaze_tracker.get_gaze_coordinates(frame, study_w, study_h)
        display_canvas = study_material.copy()

        if gaze_coords:
            g_x, g_y = gaze_coords
            
            # --- BINARY HEATMAP LOGIC ---
            if struggle_level == "high":
                heatmap_gen.add_struggle_point(g_x, g_y)
                cursor_color = (0, 0, 255) # Red for high struggle
                hud_color = (0, 0, 255)
            else:
                cursor_color = (0, 255, 0) # Green for low struggle (no heatmap point added)
                hud_color = (0, 255, 0)

            cv2.circle(display_canvas, (g_x, g_y), 15, cursor_color, -1)

        heatmap_layer = heatmap_gen.get_heatmap_overlay()
        gray_heatmap = cv2.cvtColor(heatmap_layer, cv2.COLOR_BGR2GRAY)
        mask = gray_heatmap > 0
        if mask.any():
            display_canvas[mask] = cv2.addWeighted(display_canvas[mask], 0.5, heatmap_layer[mask], 0.5, 0)

        cv2.putText(display_canvas, f"State: {current_state}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(display_canvas, f"Struggle Index: {struggle_level.upper()}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, hud_color, 2)

        cv2.imshow('StudyBuddy - Material View', display_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()