"""
StudyTracker Module (Local/Testing Version)
This script provides a localized, desktop-based tracking session using OpenCV.
It is highly useful for testing the EmotionDetector and GazeTracker 
without needing the full Flask web server running.
"""

import cv2
import time
from emotion_detector import EmotionDetector

class StudyTracker:
    def __init__(self):
        # Initialize the custom ML detector class
        self.detector = EmotionDetector()
        
        # --- Session Tracking Variables ---
        # Dictionary to accumulate the total seconds spent in each state (focused, distracted, etc.)
        self.state_durations = {}
        self.start_time = None
        self.last_frame_time = None
        self.total_time = 0

    def start_session(self):
        """
        Initiates the webcam feed, runs the calibration phase, and enters 
        the main infinite loop to process frames in real-time.
        """
        # cv2.VideoCapture(0) accesses the primary/default webcam of the system
        cap = cv2.VideoCapture(0)
        
        print("Calibrating... Please look straight at the camera with a neutral face.")
        
        # 1. Calibration Loop
        # We must wait until the EmotionDetector has gathered enough frames 
        # to establish a baseline for the user's specific facial structure.
        calibrated = False
        while not calibrated:
            ret, frame = cap.read()
            if not ret: 
                continue # Skip if the frame is corrupted or unavailable
            
            # Pass the frame to the detector; it returns True when baseline is set
            calibrated = self.detector.calibrate_off_screen(frame)
            
            # Provide visual feedback to the user during calibration
            cv2.putText(frame, "Calibrating... Look straight.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Study Tracker', frame)
            cv2.waitKey(1) # Renders the frame and waits 1ms for keyboard input
            
        print("\nCalibrated! Starting study session.")
        print("Press 'q' in the video window to end your session and generate the report.\n")
        
        # Initialize timers immediately after calibration finishes
        self.start_time = time.time()
        self.last_frame_time = self.start_time

        # 2. Main Tracking Loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Frame-Independent Time Tracking ---
            # Instead of assuming a fixed frame rate (which fluctuates depending on CPU load),
            # we calculate the exact time passed (delta_time) since the last loop iteration.
            current_time = time.time()
            delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time

            # Retrieve the current state from the AI model
            struggle_level, status = self.detector.get_struggle_index(frame)

            # Accumulate the time delta into the respective status bucket
            if status not in self.state_durations:
                self.state_durations[status] = 0.0
            self.state_durations[status] += delta_time

            # --- Heads-Up Display (HUD) ---
            # Draw real-time metrics on the OpenCV window
            color = (0, 255, 0) if struggle_level == "low" else (0, 0, 255)
            cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Struggle: {struggle_level}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Study Tracker', frame)

            # Listen for the 'q' key to gracefully exit the loop
            # 0xFF is a bitmask to ensure cross-platform compatibility with NumLock/Capslock states
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up camera resources to prevent memory leaks or hardware locks
        cap.release()
        cv2.destroyAllWindows()
        
        # 3. Generate the Final Report upon exit
        self.generate_report()

    def generate_report(self):
        """
        Calculates final session metrics (efficiency, total time) 
        and outputs them to both the console and a local text file.
        """
        self.total_time = time.time() - self.start_time
        
        # Calculate efficiency: (Time focused / Total time) * 100
        # Includes a fallback check to prevent ZeroDivisionError if exited instantly
        focused_time = self.state_durations.get("focused", 0.0)
        efficiency = (focused_time / self.total_time) * 100 if self.total_time > 0 else 0

        # Output to Console
        print("\n" + "="*50)
        print("FINAL STUDY SESSION REPORT")
        print("="*50)
        print(f"Total Study Time: {self.total_time / 60:.2f} minutes")
        print(f"Overall Efficiency: {efficiency:.1f}%")
        print("-" * 50)
        print("Detailed Breakdown:")
        
        # Sort states sequentially by longest duration
        sorted_states = sorted(self.state_durations.items(), key=lambda x: x[1], reverse=True)
        for state, duration in sorted_states:
            percentage = (duration / self.total_time) * 100
            print(f"  * {state.capitalize()}: {duration/60:.2f} mins ({percentage:.1f}%)")
        
        print("="*50)
        
        # Save to a timestamped Text File
        # Using a timestamp prevents overwriting previous session logs
        filename = f"study_report_{int(time.time())}.txt"
        with open(filename, "w") as f:
            f.write("FINAL STUDY SESSION REPORT\n")
            f.write("==========================\n")
            f.write(f"Total Study Time: {self.total_time / 60:.2f} minutes\n")
            f.write(f"Overall Efficiency: {efficiency:.1f}%\n\n")
            f.write("Detailed Breakdown:\n")
            for state, duration in sorted_states:
                f.write(f"- {state.capitalize()}: {duration/60:.2f} mins ({(duration/self.total_time)*100:.1f}%)\n")
                
        print(f"\nReport saved locally as '{filename}'")

if __name__ == "__main__":
    tracker = StudyTracker()
    tracker.start_session()