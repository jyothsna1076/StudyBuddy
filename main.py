import cv2
import numpy as np
import fitz  # PyMuPDF
import time
import sys
from emotion_detector import EmotionDetector
from heatmap_generator import HeatmapGenerator
from mouse_tracker import MouseTracker 
from calibration import CalibrationManager

# --- OPTIMIZED PDF LOADING ---
def load_pdf_page(doc, page_num):
    """Loads a SINGLE page as a normal-sized image, zero lag."""
    try:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading page {page_num}: {e}")
        return None

# --- UI HELPER ---
def draw_ui(canvas, state, level, color, mode, scroll_pos, current_page, total_pages):
    cv2.rectangle(canvas, (10, 10), (380, 190), (255, 255, 255), -1)
    cv2.rectangle(canvas, (10, 10), (380, 190), (0, 0, 0), 1)
    
    cv2.putText(canvas, f"MODE: {mode.upper()}", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(canvas, f"STATE: {str(state).upper()}", (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"STRUGGLE: {str(level).upper()}", (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(canvas, f"PAGE: {current_page + 1} / {total_pages}", (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(canvas, f"SCROLL: {scroll_pos}px", (25, 175),
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

        # 2. LOAD PDF DOC
        pdf_path = 'assets/study_material.pdf'
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if total_pages == 0:
            print(f"CRITICAL ERROR: '{pdf_path}' is empty or not found!")
            return
            
        current_page = 0
        current_doc_img = load_pdf_page(doc, current_page)
        DOC_H, DOC_W = current_doc_img.shape[:2]

        # 3. HARDWARE & TRACKING INIT
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("CRITICAL ERROR: Camera not found!")
            return

        emotion_detector = EmotionDetector()
        
        # We now keep a dictionary of heatmaps (one per page)
        heatmaps = {0: HeatmapGenerator(DOC_W, DOC_H)}

        # 4. CALIBRATION
        calibrator = CalibrationManager(None, emotion_detector) 
        calibrator.calibrate_camera_center(cap, DOC_W, min(850, DOC_H))
        
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
        
        VIEW_H = 850
        scroll_y = 0
        scroll_speed = 35 
        
        cached_heatmap = None
        heatmap_needs_update = True

        print("\n--- MAIN SESSION ACTIVE ---")
        print("Move mouse to top/bottom edges to scroll through pages. Press 'Q' to end session.")

        while cap.isOpened():
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            ret, frame = cap.read()
            if not ret: break

            current_time = time.time()
            delta_time = current_time - last_frame_time
            last_frame_time = current_time

            # --- PAGE & SCROLLING LOGIC ---
            mx, my = mouse.get_position()
            
            # Clamp viewport in case page is shorter than 850px
            actual_view_h = min(VIEW_H, DOC_H)

            if my < actual_view_h * 0.15: # Top 15%
                scroll_y -= scroll_speed
                # If we scroll past the top of the page, go to previous page
                if scroll_y < 0:
                    if current_page > 0:
                        current_page -= 1
                        current_doc_img = load_pdf_page(doc, current_page)
                        DOC_H, DOC_W = current_doc_img.shape[:2]
                        if current_page not in heatmaps:
                            heatmaps[current_page] = HeatmapGenerator(DOC_W, DOC_H)
                        scroll_y = max(0, DOC_H - actual_view_h) # Start at bottom of previous page
                        cached_heatmap = None
                        heatmap_needs_update = True
                    else:
                        scroll_y = 0 # Stuck at top of page 1
                        
            elif my > actual_view_h * 0.85: # Bottom 15%
                scroll_y += scroll_speed
                # If we scroll past the bottom of the page, go to next page
                if scroll_y > DOC_H - actual_view_h:
                    if current_page < total_pages - 1:
                        current_page += 1
                        current_doc_img = load_pdf_page(doc, current_page)
                        DOC_H, DOC_W = current_doc_img.shape[:2]
                        if current_page not in heatmaps:
                            heatmaps[current_page] = HeatmapGenerator(DOC_W, DOC_H)
                        scroll_y = 0 # Start at top of new page
                        cached_heatmap = None
                        heatmap_needs_update = True
                    else:
                        scroll_y = DOC_H - actual_view_h # Stuck at bottom of last page

            abs_doc_y = scroll_y + my
            active_heatmap = heatmaps[current_page]

            # --- AI PROCESSING ---
            if frame_count % 10 == 0:
                result = emotion_detector.get_struggle_index(frame)
                struggle_level = result[0]
                state = result[1]
            frame_count += 1

            state_durations[state] = state_durations.get(state, 0.0) + delta_time

            # --- RENDERING ---
            try:
                view_frame = current_doc_img[scroll_y : scroll_y + actual_view_h, 0 : DOC_W].copy()
                
                # Update Heatmap
                if str(struggle_level).lower() == "high":
                    active_heatmap.add_struggle_point(mx, abs_doc_y, mode=track_mode)
                    heatmap_needs_update = True 
                    color = (0, 0, 255) 
                else:
                    color = (0, 255, 0) 

                # Apply Overlay (Optimized)
                if heatmap_needs_update or cached_heatmap is None:
                    cached_heatmap = active_heatmap.get_heatmap_overlay()
                    heatmap_needs_update = False

                if cached_heatmap is not None:
                    h_slice = cached_heatmap[scroll_y : scroll_y + actual_view_h, 0 : DOC_W]
                    cv2.addWeighted(view_frame, 0.7, h_slice, 0.3, 0, view_frame)

                if track_mode == "cursor":
                    cv2.circle(view_frame, (mx, my), 15, color, 2)
                
                draw_ui(view_frame, state, struggle_level, color, track_mode, scroll_y, current_page, total_pages)
                
                cv2.imshow(WIN_NAME, view_frame)
                cv2.imshow("Webcam Feed", cv2.flip(frame, 1))

            except Exception as render_err:
                pass 

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