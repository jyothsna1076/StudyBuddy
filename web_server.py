from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import time
from emotion_detector import EmotionDetector

app = Flask(__name__)

# Initialize your UNTOUCHED AI classes here
print("Initializing AI Models...")
detector = EmotionDetector()
print("AI Models Ready.")

# Server-side session tracking
session_data = {
    "start_time": None, 
    "state_durations": {},
    "last_update": None,
    "latest_state": "Neutral",
    "latest_struggle": "Low"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        
        # Decode image
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # --- THE FIX: AUTO-CALIBRATION ---
        # If the baseline is None, it means we haven't calibrated yet.
        if detector.brow_ratio_baseline is None:
            success = detector.calibrate_off_screen(frame)
            if success:
                print("✅ Calibration Successful! Baselines set.")
            else:
                print("⚠️ Calibration failed (no face detected). Trying again next frame...")
            
            # Return early so we don't calculate struggle on the calibration frame
            return jsonify({"status": "calibrating"})
        # ---------------------------------

        # AI Logic (This now runs WITH proper baselines)
        struggle_level, state = detector.get_struggle_index(frame)

        # Update latest status for the "Check State" button
        session_data["latest_state"] = state
        session_data["latest_struggle"] = struggle_level

        # Update time tracking
        if session_data["last_update"] is not None:
            current_time = time.time()
            delta = current_time - session_data["last_update"]
            session_data["last_update"] = current_time
            
            if state not in session_data["state_durations"]:
                session_data["state_durations"][state] = 0.0
            session_data["state_durations"][state] += delta

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/start_session', methods=['POST'])
def start_session():
    # Reset the timer
    session_data["start_time"] = time.time()
    session_data["last_update"] = time.time()
    session_data["state_durations"] = {}
    
    # --- THE FIX: RESET CALIBRATION ---
    # This forces the system to grab a fresh baseline every time you hit "Start"
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
    
    return jsonify({
        "total_time_minutes": round(total_time / 60, 2),
        "efficiency": round(efficiency, 1),
        "breakdown": session_data["state_durations"]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)