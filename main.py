import mediapipe as mp
import cv2
import time
from camera import frameCapture
from predict_model import predict_rep

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

start_thresh = 0.02
reversal = 0.01
end_thresh = 0.012
calibration_delay = 5

feedback = {
    "good": "Good rep",
    "knee_valgus": "Don't let your knees cave",
    "shallow": "Go deeper",
    "shallow + knee_valgus": "Go deeper and don't let your knees cave"
}

standing_hip_y = None
standing_samples = []
start_time = time.time()
state = "standing"
current_rep = []
max_displacement = 0
record_start = 0
rep_count = 0
message = ""
color = (255, 255, 255)

for frame in frameCapture():
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        frame_data = []

        for landmark in landmarks:
            frame_data.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])

        hip_y = (landmarks[23].y + landmarks[24].y) / 2

        if standing_hip_y is None:
            if time.time() - start_time > calibration_delay:
                standing_samples.append(hip_y)
                if len(standing_samples) >= 30:
                    standing_hip_y = sum(standing_samples) / len(standing_samples)
        else:
            displacement = hip_y - standing_hip_y

            if state == "standing":
                if displacement > start_thresh:
                    state = "descending"
                    current_rep = []
                    max_displacement = displacement
            elif state == "descending":
                current_rep.append(frame_data)
                max_displacement = max(max_displacement, displacement)
                if displacement < max_displacement - reversal:
                    state = "ascending"
            elif state == "ascending":
                current_rep.append(frame_data)
                if len(current_rep) >= 5:
                    rep_count += 1
                    label = predict_rep(current_rep)['label']
                    message = feedback.get(label, label)
                    color = (0, 255, 0) if label == "good" else (0, 0, 255)
                    print(f"Rep {rep_count}: {message}")
                current_rep = []
                state = "standing"

    if standing_hip_y is None:
        cv2.putText(frame, "Stand still...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Reps: {rep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Squat Coach", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        standing_hip_y = None
        standing_samples = []
        start_time = time.time()
        state = "standing"
        current_rep = []

pose.close()
cv2.destroyAllWindows()
