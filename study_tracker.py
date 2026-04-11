import cv2
import time
from emotion_detector import EmotionDetector

class StudyTracker:
    def __init__(self):
        # Initialize the detector we built
        self.detector = EmotionDetector()
        
        # Tracking variables
        self.state_durations = {}
        self.start_time = None
        self.last_frame_time = None
        self.total_time = 0

    def start_session(self):
        cap = cv2.VideoCapture(0)
        
        print("Calibrating... Please look straight at the camera with a neutral face.")
        
        # 1. Calibration Loop
        calibrated = False
        while not calibrated:
            ret, frame = cap.read()
            if not ret: continue
            
            calibrated = self.detector.calibrate_off_screen(frame)
            
            cv2.putText(frame, "Calibrating... Look straight.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Study Tracker', frame)
            cv2.waitKey(1)
            
        print("\n✅ Calibrated! Starting study session.")
        print("Press 'q' in the video window to end your session and generate the report.\n")
        
        self.start_time = time.time()
        self.last_frame_time = self.start_time

        # 2. Main Tracking Loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate time passed since last frame
            current_time = time.time()
            delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time

            # Get status from our detector
            struggle_level, status = self.detector.get_struggle_index(frame)

            # Add the time delta to whatever status you are currently in
            if status not in self.state_durations:
                self.state_durations[status] = 0.0
            self.state_durations[status] += delta_time

            # Display info on screen
            color = (0, 255, 0) if struggle_level == "low" else (0, 0, 255)
            cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Struggle: {struggle_level}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Study Tracker', frame)

            # Quit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up camera
        cap.release()
        cv2.destroyAllWindows()
        
        # 3. Generate the Final Report
        self.generate_report()

    def generate_report(self):
        self.total_time = time.time() - self.start_time
        
        # Calculate efficiency (Time focused / Total time)
        focused_time = self.state_durations.get("focused", 0.0)
        efficiency = (focused_time / self.total_time) * 100 if self.total_time > 0 else 0

        # Print to Console
        print("\n" + "="*50)
        print("📝 FINAL STUDY SESSION REPORT")
        print("="*50)
        print(f"⏱️  Total Study Time: {self.total_time / 60:.2f} minutes")
        print(f"🎯 Overall Efficiency: {efficiency:.1f}%")
        print("-" * 50)
        print("Detailed Breakdown:")
        
        # Sort states by longest duration
        sorted_states = sorted(self.state_durations.items(), key=lambda x: x[1], reverse=True)
        for state, duration in sorted_states:
            percentage = (duration / self.total_time) * 100
            print(f"  • {state.capitalize()}: {duration/60:.2f} mins ({percentage:.1f}%)")
        
        print("="*50)
        
        # Save to Text File
        filename = f"study_report_{int(time.time())}.txt"
        with open(filename, "w") as f:
            f.write("FINAL STUDY SESSION REPORT\n")
            f.write("==========================\n")
            f.write(f"Total Study Time: {self.total_time / 60:.2f} minutes\n")
            f.write(f"Overall Efficiency: {efficiency:.1f}%\n\n")
            f.write("Detailed Breakdown:\n")
            for state, duration in sorted_states:
                f.write(f"- {state.capitalize()}: {duration/60:.2f} mins ({(duration/self.total_time)*100:.1f}%)\n")
                
        print(f"\n💾 Report saved locally as '{filename}'")

if __name__ == "__main__":
    tracker = StudyTracker()
    tracker.start_session()