"""
StudyBuddy Core Backend Server
Handles frontend communication, real-time webcam frame processing, 
state management, and dynamic PDF report generation.
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2            # OpenCV for image processing
import numpy as np    # For handling image byte arrays
import base64         # To decode images sent from the frontend via JSON
import time           # For calculating time deltas (duration of focus/distraction)
import os             # For checking file existence on the server
import fitz           # PyMuPDF library used for drawing annotations on the PDF

# Import your AI classes (Handles the Heavy Lifting for ML processing)
from emotion_detector import EmotionDetector
from gaze_tracker import GazeTracker

app = Flask(__name__)

print("Initializing AI Models...")
detector = EmotionDetector()
gaze_tracker = GazeTracker()
print("AI Models Ready.")

# Server-side session tracking
# Note for Viva: In a production app, we would use a database or Redis and map this to user IDs. 
# For this prototype, a global dictionary works perfectly to act as our in-memory data store.
session_data = {
    "start_time": None, 
    "state_durations": {},
    "last_update": None,
    "latest_state": "Neutral",
    "latest_struggle": "Low",
    "struggle_zones": {}, 
    "focus_zones": {},    
    "pdf_path": None,
    "tracking_mode": "page",
    # --- NEW: Page Level Tracking ---
    "page_stats": {}, 
    "last_active_page": None 
}

@app.route('/')
def index():
    """Serves the main HTML interface."""
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """
    Receives the PDF file from the user via multipart/form-data.
    Saves it locally so the server can process and annotate it later.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
        
    if file:
        file_path = "temp_study_material.pdf"
        file.save(file_path)
        session_data["pdf_path"] = file_path
        return jsonify({"status": "success", "message": "PDF uploaded to server"})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    The core loop endpoint. The frontend hits this multiple times a second.
    It decodes the image, runs the AI model, calculates the time spent in the 
    current state, and maps it to the exact PDF page/line coordinates.
    """
    try:
        data = request.json
        
        # 1. Decode the base64 image payload into an OpenCV compatible format
        image_data = data['image'].split(',')[1]
        mode = data.get('mode', 'page')
        active_page = data.get('active_page', 1)
        relative_y_pct = data.get('relative_y_pct', 0.0)
        
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 2. Calibration Check
        if detector.brow_ratio_baseline is None:
            detector.calibrate_off_screen(frame)
            return jsonify({"status": "calibrating"})

        # 3. AI Processing: Extract state and struggle metrics from the frame
        struggle_level, state = detector.get_struggle_index(frame)
        session_data["latest_state"] = state
        session_data["latest_struggle"] = struggle_level

        # 4. Time Calculation: Determine how much time has passed since the last frame
        if session_data["last_update"] is not None:
            current_time = time.time()
            delta = current_time - session_data["last_update"]
            session_data["last_update"] = current_time
            
            # Update global state durations (Total study time vs distracted time)
            if state not in session_data["state_durations"]:
                session_data["state_durations"][state] = 0.0
            session_data["state_durations"][state] += delta

            # --- Page-Level Statistics Accumulation ---
            if active_page not in session_data["page_stats"]:
                session_data["page_stats"][active_page] = {"focus_time": 0.0, "distraction_time": 0.0, "visits": 0}
            
            # Track if the user scrolled to a new page
            if session_data["last_active_page"] != active_page:
                session_data["page_stats"][active_page]["visits"] += 1
                session_data["last_active_page"] = active_page

            # Credit time to the specific page based on AI classification
            if state == "focused":
                session_data["page_stats"][active_page]["focus_time"] += delta
            elif struggle_level == "high" or state == "distracted":
                session_data["page_stats"][active_page]["distraction_time"] += delta

            # --- PERCENTAGE TRACKING (For Cursor/Exact Line tracking mode) ---
            # We group coordinates into 10% buckets (0.1, 0.2, etc.) to group nearby lines together.
            if struggle_level == "high" or state == "focused":
                pct_bucket = round(relative_y_pct * 10) / 10.0
                zone_key = f"{active_page}_{pct_bucket}"
                
                if struggle_level == "high":
                    if zone_key not in session_data["struggle_zones"]:
                        session_data["struggle_zones"][zone_key] = 0.0
                    session_data["struggle_zones"][zone_key] += delta
                    
                if state == "focused":
                    if zone_key not in session_data["focus_zones"]:
                        session_data["focus_zones"][zone_key] = 0.0
                    session_data["focus_zones"][zone_key] += delta

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
    
@app.route('/start_session', methods=['POST'])
def start_session():
    """
    Resets all variables and timers. Called when the user clicks 'Start Session'.
    Ensures baselines are cleared so a new calibration is forced.
    """
    data = request.json or {}
    session_data["tracking_mode"] = data.get("tracking_mode", "page")
    
    session_data["start_time"] = time.time()
    session_data["last_update"] = time.time()
    session_data["state_durations"] = {}
    session_data["struggle_zones"] = {} 
    session_data["focus_zones"] = {} 
    
    # Reset page stats and tracking markers
    session_data["page_stats"] = {}
    session_data["last_active_page"] = None
    
    # Nullify AI baselines to trigger recalibration for lighting/position changes
    detector.brow_ratio_baseline = None 
    detector.ear_baseline = None
    detector.mar_baseline = None
    
    return jsonify({"status": "Session Started"})

@app.route('/get_current_state', methods=['GET'])
def get_current_state():
    """Polling endpoint used by the frontend's Live Monitor modal."""
    return jsonify({
        "state": session_data["latest_state"],
        "struggle": session_data["latest_struggle"]
    })

@app.route('/get_report', methods=['GET'])
def get_report():
    """
    The final compilation step. Calculates overall efficiency metrics, 
    filters out the 'noise' (data below thresholds), and uses PyMuPDF (fitz)
    to draw transparent colored boxes on the original PDF document.
    """
    if session_data["start_time"] is None:
        return jsonify({"error": "Session not started"})

    total_time = time.time() - session_data["start_time"]
    focused_time = session_data["state_durations"].get("focused", 0.0)
    
    # Prevent divide-by-zero errors
    efficiency = (focused_time / total_time) * 100 if total_time > 0 else 0
    mode = session_data.get("tracking_mode", "page")
    
    # --- PARSE LINE ZONES (For Cursor Mode) ---
    heavy_struggle_areas = []
    # Only flag areas where distraction/struggle exceeded 10 seconds to avoid micro-expressions
    for zone_key, duration in session_data["struggle_zones"].items():
        if duration >= 10.0: 
            page_num_str, pct_str = zone_key.split('_')
            heavy_struggle_areas.append({
                "estimated_page": int(page_num_str),
                "relative_pct": float(pct_str),
                "duration_seconds": round(duration, 1)
            })

    heavy_focus_areas = []
    # Sort zones by duration and grab only the top 5 most focused lines
    sorted_focus_zones = sorted(session_data["focus_zones"].items(), key=lambda x: x[1], reverse=True)
    for zone_key, duration in sorted_focus_zones[:5]:
        if duration >= 10.0:
            page_num_str, pct_str = zone_key.split('_')
            heavy_focus_areas.append({
                "estimated_page": int(page_num_str),
                "relative_pct": float(pct_str),
                "duration_seconds": round(duration, 1)
            })

    # --- PDF GENERATION USING PyMuPDF (fitz) ---
    report_pdf_url = None

    if session_data.get("pdf_path") and os.path.exists(session_data["pdf_path"]):
        try:
            doc = fitz.open(session_data["pdf_path"])
            
            if mode == "line":
                # === LINE MODE (Exact Cursor Tracking) ===
                def highlight_areas(areas, color):
                    """Helper function to draw horizontal highlight bands over exact lines"""
                    for area in areas:
                        # fitz pages are 0-indexed, our tracking is 1-indexed
                        page_idx = area["estimated_page"] - 1
                        if 0 <= page_idx < len(doc):
                            page = doc[page_idx]
                            
                            # Map the 0.0 - 1.0 percentage to actual PDF pixel coordinates
                            pdf_y = area["relative_pct"] * page.rect.height
                            pdf_band_height = page.rect.height * 0.10 # 10% vertical band thickness
                            
                            # Define the rectangle (x0, y0, x1, y1)
                            rect = fitz.Rect(0, pdf_y, page.rect.width, pdf_y + pdf_band_height)
                                
                            annot = page.add_rect_annot(rect)
                            annot.set_colors(stroke=color, fill=color)
                            annot.set_opacity(0.2) # Make it transparent so text is readable
                            annot.update()

                highlight_areas(heavy_struggle_areas, (1, 0, 0)) # Red Lines for struggle
                highlight_areas(heavy_focus_areas, (0, 1, 0))    # Green Lines for focus

            else:
                # === PAGE MODE (General Gaze Tracking) ===
                red_pages = []
                for page_num, stats in session_data["page_stats"].items():
                    if stats["distraction_time"] > 30.0: # Threshold for a 'distracted' page
                        red_pages.append(page_num)
                        
                # Sort pages by focus time and grab the top 5
                sorted_focus = sorted(session_data["page_stats"].items(), key=lambda x: x[1]["focus_time"], reverse=True)
                green_pages = [page_num for page_num, stats in sorted_focus if stats["focus_time"] > 0][:5]
                
                # Identify the page the user flipped back to the most
                blue_page = None
                if session_data["page_stats"]:
                    most_visited = max(session_data["page_stats"].items(), key=lambda x: x[1]["visits"])
                    if most_visited[1]["visits"] > 1:
                        blue_page = most_visited[0]

                # Apply page-level full-page overlays
                for page_idx in range(len(doc)):
                    page_num = page_idx + 1
                    page = doc[page_idx]
                    
                    if page_num == blue_page:
                        annot = page.add_rect_annot(page.rect)
                        annot.set_colors(stroke=(0, 0, 1)) # Blue border
                        annot.set_border(width=10)
                        annot.update()

                    if page_num in red_pages:
                        annot = page.add_rect_annot(page.rect)
                        annot.set_colors(stroke=(1, 0, 0), fill=(1, 0, 0)) # Red tint
                        annot.set_opacity(0.15)
                        annot.update()

                    if page_num in green_pages:
                        annot = page.add_rect_annot(page.rect)
                        annot.set_colors(stroke=(0, 1, 0), fill=(0, 1, 0)) # Green tint
                        annot.set_opacity(0.15)
                        annot.update()
                        
            output_path = "Report_StudyBuddy.pdf"
            doc.save(output_path)
            doc.close()
            
            # Pass the URL endpoint to the frontend so the user can download it
            report_pdf_url = "/download_report_pdf"
        except Exception as e:
            print(f"Error generating PDF: {e}")

    return jsonify({
        "total_time_minutes": round(total_time / 60, 2),
        "efficiency": round(efficiency, 1),
        "breakdown": session_data["state_durations"],
        "hardest_sections": heavy_struggle_areas,
        "report_pdf_url": report_pdf_url,
        "mode": mode  
    })

@app.route('/download_report_pdf', methods=['GET'])
def download_report_pdf():
    """Endpoint that actually serves the saved PDF binary back to the user's browser."""
    return send_file("Report_StudyBuddy.pdf", as_attachment=True)

if __name__ == '__main__':
    # Run the Flask app on localhost, port 5000
    app.run(debug=True, port=5000)