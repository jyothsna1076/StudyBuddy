from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import time
import os
import fitz  

# Import your AI classes
from emotion_detector import EmotionDetector
from gaze_tracker import GazeTracker

app = Flask(__name__)

print("Initializing AI Models...")
detector = EmotionDetector()
gaze_tracker = GazeTracker()
print("AI Models Ready.")

# Server-side session tracking
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
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
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
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        
        mode = data.get('mode', 'page')
        active_page = data.get('active_page', 1)
        relative_y_pct = data.get('relative_y_pct', 0.0)
        
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if detector.brow_ratio_baseline is None:
            detector.calibrate_off_screen(frame)
            return jsonify({"status": "calibrating"})

        struggle_level, state = detector.get_struggle_index(frame)
        session_data["latest_state"] = state
        session_data["latest_struggle"] = struggle_level

        if session_data["last_update"] is not None:
            current_time = time.time()
            delta = current_time - session_data["last_update"]
            session_data["last_update"] = current_time
            
            if state not in session_data["state_durations"]:
                session_data["state_durations"][state] = 0.0
            session_data["state_durations"][state] += delta

            # --- NEW: Page-Level Statistics (Visits, Focus Time, Distraction Time) ---
            if active_page not in session_data["page_stats"]:
                session_data["page_stats"][active_page] = {"focus_time": 0.0, "distraction_time": 0.0, "visits": 0}
            
            if session_data["last_active_page"] != active_page:
                session_data["page_stats"][active_page]["visits"] += 1
                session_data["last_active_page"] = active_page

            if state == "focused":
                session_data["page_stats"][active_page]["focus_time"] += delta
            elif struggle_level == "high" or state == "distracted":
                session_data["page_stats"][active_page]["distraction_time"] += delta
            # -------------------------------------------------------------------------

            # --- PERCENTAGE TRACKING (Kept intact) ---
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
    data = request.json or {}
    session_data["tracking_mode"] = data.get("tracking_mode", "page")
    
    session_data["start_time"] = time.time()
    session_data["last_update"] = time.time()
    session_data["state_durations"] = {}
    session_data["struggle_zones"] = {} 
    session_data["focus_zones"] = {} 
    
    # --- NEW: Reset page stats ---
    session_data["page_stats"] = {}
    session_data["last_active_page"] = None
    
    detector.brow_ratio_baseline = None 
    detector.ear_baseline = None
    detector.mar_baseline = None
    
    return jsonify({"status": "Session Started"})

@app.route('/get_current_state', methods=['GET'])
def get_current_state():
    return jsonify({
        "state": session_data["latest_state"],
        "struggle": session_data["latest_struggle"]
    })

@app.route('/get_report', methods=['GET'])
def get_report():
    if session_data["start_time"] is None:
        return jsonify({"error": "Session not started"})

    total_time = time.time() - session_data["start_time"]
    focused_time = session_data["state_durations"].get("focused", 0.0)
    efficiency = (focused_time / total_time) * 100 if total_time > 0 else 0
    
    # Process the new percentage dictionary keys instead of pixel math
    heavy_struggle_areas = []
    for zone_key, duration in session_data["struggle_zones"].items():
        if duration >= 10.0:
            page_num_str, pct_str = zone_key.split('_')
            heavy_struggle_areas.append({
                "estimated_page": int(page_num_str),
                "relative_pct": float(pct_str),
                "duration_seconds": round(duration, 1)
            })

    heavy_focus_areas = []
    for zone_key, duration in session_data["focus_zones"].items():
        if duration >= 30.0:
            page_num_str, pct_str = zone_key.split('_')
            heavy_focus_areas.append({
                "estimated_page": int(page_num_str),
                "relative_pct": float(pct_str),
                "duration_seconds": round(duration, 1)
            })

    # --- PDF GENERATION WITH EXACT LINE PERCENTAGES ---
    # --- NEW: PDF GENERATION WITH 3 COLOR RULES ---
    report_pdf_url = None

    if session_data.get("pdf_path") and os.path.exists(session_data["pdf_path"]):
        try:
            doc = fitz.open(session_data["pdf_path"])
            
            # Calculate the 3 target groups
            red_pages = []
            for page_num, stats in session_data["page_stats"].items():
                if stats["distraction_time"] > 30.0:
                    red_pages.append(page_num)
                    
            sorted_focus = sorted(session_data["page_stats"].items(), key=lambda x: x[1]["focus_time"], reverse=True)
            green_pages = [page_num for page_num, stats in sorted_focus if stats["focus_time"] > 0][:5]
            
            blue_page = None
            if session_data["page_stats"]:
                most_visited = max(session_data["page_stats"].items(), key=lambda x: x[1]["visits"])
                if most_visited[1]["visits"] > 1: # Make sure they actually visited it more than once
                    blue_page = most_visited[0]

            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                page = doc[page_idx]
                
                # 3. BLUE: Most Revisited (Draws a thick blue border)
                if page_num == blue_page:
                    annot = page.add_rect_annot(page.rect)
                    annot.set_colors(stroke=(0, 0, 1))
                    annot.set_border(width=10)
                    annot.update()

                # 1. RED: Distraction > 30s (Fills the page red)
                if page_num in red_pages:
                    annot = page.add_rect_annot(page.rect)
                    annot.set_colors(stroke=(1, 0, 0), fill=(1, 0, 0))
                    annot.set_opacity(0.15)
                    annot.update()

                # 2. GREEN: Top 5 Focus (Fills the page green)
                if page_num in green_pages:
                    annot = page.add_rect_annot(page.rect)
                    annot.set_colors(stroke=(0, 1, 0), fill=(0, 1, 0))
                    annot.set_opacity(0.15)
                    annot.update()
                    
            output_path = "Report_StudyBuddy.pdf"
            doc.save(output_path)
            doc.close()
            
            report_pdf_url = "/download_report_pdf"
        except Exception as e:
            print(f"Error generating PDF: {e}")

    return jsonify({
        "total_time_minutes": round(total_time / 60, 2),
        "efficiency": round(efficiency, 1),
        "breakdown": session_data["state_durations"],
        "hardest_sections": heavy_struggle_areas,
        "report_pdf_url": report_pdf_url 
    })

@app.route('/download_report_pdf', methods=['GET'])
def download_report_pdf():
    return send_file("Report_StudyBuddy.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)