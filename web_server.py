from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import time

# Import your AI classes
from emotion_detector import EmotionDetector
from gaze_tracker import GazeTracker

app = Flask(__name__)

# Initialize your AI classes here
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
    "struggle_zones": {}  # NEW: Tracks seconds spent struggling at specific Y-coordinates
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        
        # NEW: Get screen and scroll data from the frontend (defaulting to 0 if not sent)
        scroll_y = data.get('scroll_y', 0)
        screen_w = data.get('screen_w', 1920)
        screen_h = data.get('screen_h', 1080)
        
        # Decode image
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # --- AUTO-CALIBRATION ---
        if detector.brow_ratio_baseline is None:
            success = detector.calibrate_off_screen(frame)
            if success:
                print("✅ Calibration Successful! Baselines set.")
                
                # Calibrate gaze tracker center as well
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.holistic.process(rgb_frame)
                if results.face_landmarks:
                    # Quick calibration for gaze based on holistic landmarks
                    iris_x = (results.face_landmarks.landmark[468].x + results.face_landmarks.landmark[473].x) / 2
                    iris_y = (results.face_landmarks.landmark[468].y + results.face_landmarks.landmark[473].y) / 2
                    gaze_tracker.center_rel_x = iris_x - results.face_landmarks.landmark[6].x
                    gaze_tracker.center_rel_y = iris_y - results.face_landmarks.landmark[6].y
            else:
                print("⚠️ Calibration failed (no face detected). Trying again next frame...")
            
            return jsonify({"status": "calibrating"})
        # ---------------------------------

        # AI Logic
        struggle_level, state = detector.get_struggle_index(frame)

        # Update latest status for the "Check State" button
        session_data["latest_state"] = state
        session_data["latest_struggle"] = struggle_level

        # Update time and location tracking
        if session_data["last_update"] is not None:
            current_time = time.time()
            delta = current_time - session_data["last_update"]
            session_data["last_update"] = current_time
            
            if state not in session_data["state_durations"]:
                session_data["state_durations"][state] = 0.0
            session_data["state_durations"][state] += delta

            # --- NEW: MAP STRUGGLE TO DOCUMENT LINES ---
            if struggle_level == "high":
                gaze_coords = gaze_tracker.get_gaze_coordinates(frame, screen_w, screen_h)
                if gaze_coords:
                    gaze_x, gaze_y = gaze_coords
                    
                    # Absolute Y is where you are looking + how far down the page you scrolled
                    absolute_y = gaze_y + scroll_y
                    
                    # Group the page into "buckets" of 150 pixels (roughly 4-5 lines of text)
                    line_bucket = int((absolute_y // 150) * 150)
                    
                    if line_bucket not in session_data["struggle_zones"]:
                        session_data["struggle_zones"][line_bucket] = 0.0
                    session_data["struggle_zones"][line_bucket] += delta

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/start_session', methods=['POST'])
def start_session():
    # Reset the timer and zones
    session_data["start_time"] = time.time()
    session_data["last_update"] = time.time()
    session_data["state_durations"] = {}
    session_data["struggle_zones"] = {} # Reset zones for new session
    
    # Force fresh baseline calibration
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
    
    # --- NEW: EXTRACT HEAVY STRUGGLE AREAS ---
    heavy_struggle_areas = []
    
    # Sort zones by Y coordinate (top to bottom of page)
    for y_pos in sorted(session_data["struggle_zones"].keys()):
        duration = session_data["struggle_zones"][y_pos]
        
        # Only report if they struggled in this specific section for more than 45 seconds
        if duration >= 10.0:
            # Estimate which "Page" this is on (Assuming 900px average height)
            page_num = (y_pos // 900) + 1 
            heavy_struggle_areas.append({
                "document_pixel_y": y_pos,
                "estimated_page": page_num,
                "duration_seconds": round(duration, 1)
            })
    
    return jsonify({
        "total_time_minutes": round(total_time / 60, 2),
        "efficiency": round(efficiency, 1),
        "breakdown": session_data["state_durations"],
        "hardest_sections": heavy_struggle_areas # Adds the zones to your report!
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)