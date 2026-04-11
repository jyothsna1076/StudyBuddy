import cv2
from gaze_tracker import GazeTracker
from emotion_detector import EmotionDetector
from heatmap_generator import HeatmapGenerator
from calibration import CalibrationManager

def draw_ui(canvas, state, level, color):
    cv2.putText(canvas, f"State: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(canvas, f"Struggle: {level}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main():
    study_material = cv2.imread('assets/study_material.png')
    if study_material is None:
        print("Missing study material")
        return

    h, w = study_material.shape[:2]

    cap = cv2.VideoCapture(0)

    gaze_tracker = GazeTracker()
    emotion_detector = EmotionDetector()
    heatmap = HeatmapGenerator(w, h)

    # NEW CLEAN CALIBRATION
    calibrator = CalibrationManager(gaze_tracker, emotion_detector)
    calibrator.run_full_calibration(cap, w, h)

    print("System Ready. Press Q to quit.")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam", cv2.flip(frame, 1))

        # Emotion (optimized)
        if frame_count % 10 == 0:
            struggle_level, state = emotion_detector.get_struggle_index(frame)

        frame_count += 1

        gaze = gaze_tracker.get_gaze_coordinates(frame, w, h)
        canvas = study_material.copy()

        if gaze:
            x, y = gaze

            if struggle_level == "high":
                heatmap.add_struggle_point(x, y)
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.circle(canvas, (x, y), 15, color, -1)

        # Heatmap overlay
        overlay = heatmap.get_heatmap_overlay()

        if overlay is not None:
            gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            mask = gray > 0

            if mask.any():
                canvas[mask] = cv2.addWeighted(
                    canvas[mask], 0.5,
                    overlay[mask], 0.5,
                    0
                )

        # UI (clean)
        draw_ui(canvas, state, struggle_level.upper(), color)

        cv2.imshow("StudyBuddy", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()