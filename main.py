import cv2
import numpy as np
import fitz  # PyMuPDF
import time
import sys
from emotion_detector import EmotionDetector
from heatmap_generator import HeatmapGenerator
from mouse_tracker import MouseTracker 
from calibration import CalibrationManager

# --- PDF LOADING ---
def load_full_document(pdf_path):
    """Stacks all PDF pages into one giant vertical image."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        print(f"Processing {len(doc)} pages...")
        for i in range(len(doc)):
            page = doc.load_page(i)
            # 2.0x scale provides a good balance of clarity and performance
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            pages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        full_img = np.vstack(pages)
        return full_img
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

# --- UI HELPER ---
def draw_ui(canvas, state, level, color, mode, scroll_pos):
    # Semi-transparent background box for the UI
    cv2.rectangle(canvas, (10, 10), (380, 160), (255, 255, 255), -1)
    cv2.rectangle(canvas, (10, 10), (380, 160), (0, 0, 0), 1)
    
    cv2.putText(canvas, f"MODE: {mode.upper()}", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(canvas, f"STATE: {str(state).upper()}", (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"STRUGGLE: {str(level).upper()}", (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(canvas, f"SCROLL: {scroll_pos}px", (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

def main():
    try:
        print("\n" + "="*30)
        print("STUDYBUDDY BOOT SEQUENCE")
        print("="*30)

        # 1. SELECTING MODE
        print("ACTION REQUIRED: Select mode in terminal...")
        choice = input("Selection (1 for Cursor/Line, 2 for Gaze/Page): ").strip()
        track_mode = "cursor" if choice == "1" else "gaze"

        # 2. LOAD PDF
        pdf_path = 'assets/study_material.pdf'
        full_doc = load_full_document(pdf_path)
        if full_doc is None:
            print(f"CRITICAL ERROR: '{pdf_path}' not found!")
            input("Press Enter to Exit...")
            return
        
        DOC_H, DOC_W = full_doc.shape[:2]

        # 3. HARDWARE & TRACKING INIT
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("CRITICAL ERROR: Camera not found!")
            return

        emotion_detector = EmotionDetector()
        heatmap = HeatmapGenerator(DOC_W, DOC_H)

        # 4. CALIBRATION
        calibrator = CalibrationManager(None, emotion_detector) 
        calibrator.calibrate_camera_center(cap, DOC_W, 850)
        
        cv2.destroyAllWindows()
        WIN_NAME = "StudyBuddy - Reading"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        mouse = MouseTracker(WIN_NAME)

        # --- Session Tracking Variables ---
        state_durations = {}
        start_time = time.time()
        last_frame_time = start_time
        
        frame_count = 0
        struggle_level = "low"
        state = "neutral"
        color = (0, 255, 0)
        scroll_y = 0
        VIEW_H = 850
        scroll_speed = 35 # Faster scroll for stacked pages

        print("\n--- MAIN SESSION ACTIVE ---")
        print("Move mouse to top/bottom edges to scroll. Press 'Q' to end session.")

        while cap.isOpened():
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            ret, frame = cap.read()
            if not ret: break

            # --- TIME TRACKING ---
            current_time = time.time()
            delta_time = current_time - last_frame_time
            last_frame_time = current_time

            # --- INPUT & SCROLLING ---
            mx, my = mouse.get_position()
            if my < VIEW_H * 0.15: # Top 15%
                scroll_y = max(0, scroll_y - scroll_speed)
            elif my > VIEW_H * 0.85: # Bottom 15%
                scroll_y = min(DOC_H - VIEW_H, scroll_y + scroll_speed)
            
            abs_doc_y = scroll_y + my

            # --- AI PROCESSING ---
            if frame_count % 10 == 0:
                result = emotion_detector.get_struggle_index(frame)
                struggle_level = result[0]
                state = result[1]
            frame_count += 1

            # --- RECORD DURATION ---
            state_durations[state] = state_durations.get(state, 0.0) + delta_time

            # --- RENDERING ---
            try:
                # 1. Extract Viewport from full doc
                view_frame = full_doc[scroll_y : scroll_y + VIEW_H, 0 : DOC_W].copy()
                
                # 2. Update Heatmap logic
               # --- UPDATE HEATMAP ---
                if str(struggle_level).lower() == "high":
                    # NEW: Pass scroll_y and VIEW_H to ensure only the visible page turns red
                    intensity_val = 1.0 if track_mode == "cursor" else 0.5
                    heatmap.add_struggle_point(
                        mx, 
                        abs_doc_y, 
                        mode=track_mode, 
                        scroll_y=scroll_y, 
                        view_h=VIEW_H
                    )
                    color = (0, 0, 255) # Red

                # 3. Apply Heatmap Overlay
                full_heatmap = heatmap.get_heatmap_overlay()
                if full_heatmap is not None:
                    h_slice = full_heatmap[scroll_y : scroll_y + VIEW_H, 0 : DOC_W]
                    cv2.addWeighted(view_frame, 0.7, h_slice, 0.3, 0, view_frame)

                # 4. Display UI and Cursor
                if track_mode == "cursor":
                    cv2.circle(view_frame, (mx, my), 15, color, 2)
                
                draw_ui(view_frame, state, struggle_level, color, track_mode, scroll_y)
                
                cv2.imshow(WIN_NAME, view_frame)
                cv2.imshow("Webcam Feed", cv2.flip(frame, 1))

            except Exception as render_err:
                pass # Silently handle minor clipping during fast scrolls

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- SESSION REPORT GENERATION ---
        total_time = time.time() - start_time
        focused_time = state_durations.get("focused", 0.0)
        efficiency = (focused_time / total_time) * 100 if total_time > 0 else 0

        print("\n" + "="*50)
        print("📝 FINAL STUDY SESSION REPORT")
        print("="*50)
        print(f"⏱️  Total Session: {total_time / 60:.2f} minutes")
        print(f"🎯 Efficiency: {efficiency:.1f}%")
        print("-" * 50)
        
        sorted_states = sorted(state_durations.items(), key=lambda item: item[1], reverse=True)
        for s, duration in sorted_states:
            percentage = (duration / total_time) * 100
            print(f" • {str(s).capitalize()}: {duration/60:.2f} mins ({percentage:.1f}%)")

        # Save to file
        filename = f"study_report_{int(time.time())}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("FINAL STUDY SESSION REPORT\n")
            f.write("==========================\n")
            f.write(f"Mode Used: {track_mode.upper()}\n")
            f.write(f"Total Study Time: {total_time / 60:.2f} minutes\n")
            f.write(f"Overall Efficiency: {efficiency:.1f}%\n\n")
            f.write("Detailed Breakdown:\n")
            for s, duration in sorted_states:
                f.write(f"- {str(s).capitalize()}: {duration/60:.2f} mins ({(duration/total_time)*100:.1f}%)\n")
        
        print(f"\n💾 Report saved as '{filename}'")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
    finally:
        if 'cap' in locals(): cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()