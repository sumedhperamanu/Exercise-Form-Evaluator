import mediapipe as mp
import cv2
import time
from camera import frameCapture
from predict_model import predict_rep
from collections import deque

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

start_thresh = 0.02
reversal = 0.01
end_thresh = 0.012
calibration_delay = 3

feedback = {
    "good": "Good rep",
    "knee_valgus": "Don't let your knees cave",
    "shallow": "Go deeper",
    "shallow + knee_valgus": "Go deeper and don't let your knees cave"
}

standing_hip_y = None

recent_hips = deque(maxlen=20)
rep_start = 0

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

        recent_hips.append(hip_y)
        still = len(recent_hips) == 20 and max(recent_hips) - min(recent_hips) < 0.006

        if standing_hip_y is None:
            if time.time() - start_time > calibration_delay and still:
                standing_hip_y = sum(recent_hips) / len(recent_hips)
        else:
            displacement = hip_y - standing_hip_y

            if state in ("descending", "ascending") and still and displacement < 0.05:
                if time.perf_counter() - rep_start > 2.0:
                    standing_hip_y = sum(recent_hips) / len(recent_hips)
                    state, current_rep = "standing", []
                    message = "Recalibrated"
                    color = (0, 255, 0)
                else:
                    message = "Stand still, camera is recalibrating"
                    color = (0, 255, 255)
            elif state == "standing":
                if displacement > start_thresh:
                    state = "descending"
                    current_rep = []
                    max_displacement = displacement
                    rep_start = time.perf_counter()

            elif state == "descending":
                current_rep.append(frame_data)
                max_displacement = max(max_displacement, displacement)
                if displacement < max_displacement - reversal:
                    state = "ascending"
            elif state == "ascending":
                current_rep.append(frame_data)
                if displacement < end_thresh:
                    if len(current_rep) >= 7:
                        rep_count += 1
                        label = predict_rep(current_rep)['label']
                        message = feedback.get(label, label)
                        color = (0, 255, 0) if label == "good" else (0, 0, 255)
                        print(f"Rep {rep_count}: {message}")
                    else:
                        message = "Rep too short; extend your reps"
                        color = (0, 165, 255)
                        print(message)

                    current_rep = []
                    state = "standing"

    if standing_hip_y is None:
        cv2.putText(frame, "Stand still...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Reps: {rep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
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

        recent_hips.clear()

pose.close()
cv2.destroyAllWindows()
