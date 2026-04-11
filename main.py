import cv2
import numpy as np
import time  # Added for time tracking
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

    # 2. Initialize Hardware/AI
    cap = cv2.VideoCapture(0)
    emotion_detector = EmotionDetector()
    heatmap = HeatmapGenerator(w, h)

    # 3. Calibration Phase
    calibrator = CalibrationManager(None, emotion_detector) 
    calibrator.calibrate_camera_center(cap, w, h)
    
    # CRITICAL: Clean up calibration window but immediately 
    # prepare the StudyBuddy window for the mouse tracker
    cv2.destroyAllWindows()
    cv2.namedWindow(WIN_MATERIAL) 
    mouse = MouseTracker(WIN_MATERIAL)

    print("System Ready. Mouse tracking active. Press Q or click [X] to quit.")

    # --- Added Tracking Variables ---
    state_durations = {}
    start_time = time.time()
    last_frame_time = start_time

    frame_count = 0
    struggle_level = "low"
    state = "neutral"
    color = (0, 255, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Added Time Delta Logic ---
        current_time = time.time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time

        # AI logic every 10 frames
        if frame_count % 10 == 0:
            result = emotion_detector.get_struggle_index(frame)
            struggle_level = result[0]
            state = result[1]
        
        frame_count += 1

        # --- Record State Duration ---
        if state not in state_durations:
            state_durations[state] = 0.0
        state_durations[state] += delta_time

        # GET MOUSE POSITION (updates via callback in MouseTracker)
        x, y = mouse.get_position()
        
        # Clamp to image boundaries
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        canvas = study_material.copy()

        # Update Heatmap if struggle is HIGH
        if str(struggle_level).lower() == "high":
            # This calls your heatmap.add_struggle_point logic
            heatmap.add_struggle_point(x, y)
            color = (0, 0, 255) # Red
        else:
            color = (0, 255, 0) # Green

        # Draw the "Cursor" circle (Always follows mouse)
        cv2.circle(canvas, (x, y), 15, color, -1)

        # Apply Heatmap Overlay from your HeatmapGenerator
        overlay = heatmap.get_heatmap_overlay()
        if overlay is not None:
            # We only overlay non-black pixels from the Jet colormap
            gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            mask = gray > 0
            if mask.any():
                canvas[mask] = cv2.addWeighted(canvas[mask], 0.5, overlay[mask], 0.5, 0)

        # UI and Display
        draw_ui(canvas, state, struggle_level, color)
        cv2.imshow(WIN_WEBCAM, cv2.flip(frame, 1))
        cv2.imshow(WIN_MATERIAL, canvas)

        # Exit Logic
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Proper [X] button handling
        if cv2.getWindowProperty(WIN_MATERIAL, cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty(WIN_WEBCAM, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- Added Final Report Generation ---
    total_time = time.time() - start_time
    focused_time = state_durations.get("focused", 0.0)
    efficiency = (focused_time / total_time) * 100 if total_time > 0 else 0

    print("\n" + "="*50)
    print("📝 FINAL STUDY SESSION REPORT")
    print("="*50)
    print(f"⏱️  Total Study Time: {total_time / 60:.2f} minutes")
    print(f"🎯 Overall Efficiency: {efficiency:.1f}%")
    print("-" * 50)
    print("Detailed Breakdown:")
    
    sorted_states = sorted(state_durations.items(), key=lambda item: item[1], reverse=True)
    for s, duration in sorted_states:
        percentage = (duration / total_time) * 100
        print(f"  • {str(s).capitalize()}: {duration/60:.2f} mins ({percentage:.1f}%)")
    print("="*50)
    
    filename = f"study_report_{int(time.time())}.txt"
    with open(filename, "w") as f:
        f.write("FINAL STUDY SESSION REPORT\n")
        f.write("==========================\n")
        f.write(f"Total Study Time: {total_time / 60:.2f} minutes\n")
        f.write(f"Overall Efficiency: {efficiency:.1f}%\n\n")
        f.write("Detailed Breakdown:\n")
        for s, duration in sorted_states:
            f.write(f"- {str(s).capitalize()}: {duration/60:.2f} mins ({(duration/total_time)*100:.1f}%)\n")
            
    print(f"\n💾 Report saved locally as '{filename}'")

if __name__ == "__main__":
    main()