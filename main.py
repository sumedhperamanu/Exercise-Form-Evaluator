import mediapipe as mp
import cv2
import time
from camera import frameCapture
from predict_model import predict_rep
import requests
from dotenv import load_dotenv
import os
import threading

load_dotenv()

BASE_URL = "https://api.thingspeak.com/update"
THINGSPEAK_API_KEY = os.getenv("THINGSPEAK_API_KEY")

UPLOADS_ENABLED = bool(THINGSPEAK_API_KEY)
if not UPLOADS_ENABLED:
    print("Warning: THINGSPEAK_API_KEY not set, uploads disabled for this run.")

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

total_rep_count = 0
good_rep_count = 0

message = ""
color = (255, 255, 255)

UPLOAD_INTERVAL_S = 20
last_request_time_s = time.time()

upload_thread = None


def send_data_to_thingspeak(field_name, field_value):
    payload = {
        "api_key": THINGSPEAK_API_KEY,
        field_name: field_value
    }

    try:
        # Send data via HTTP POST
        response = requests.post(BASE_URL, data=payload, timeout=5)
        
        # A status code of 200 and an entry ID > 0 means success
        if response.status_code == 200 and int(response.text) > 0:
            print(f"Success! Entry ID: {response.text} | Value: {field_value}")
            return True
        else:
            print(f"Failed to update. Response: {response.text}")
            
    except Exception as e:
        print(f"Error connecting to ThingSpeak: {e}")

    return False


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
                if displacement < end_thresh:
                    if len(current_rep) >= 5:
                        total_rep_count += 1
                        label = predict_rep(current_rep)['label']
                        message = feedback.get(label, label)
                        color = (0, 255, 0) if label == "good" else (0, 0, 255)
                        if label == "good": good_rep_count += 1
                        print(f"Rep {total_rep_count}: {message}")
                    else:
                        message = "Rep too short; extend your reps"
                        color = (0, 165, 255)
                        print(message)

                    current_rep = []
                    state = "standing"

    if standing_hip_y is None:
        cv2.putText(frame, "Stand still...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Reps: {total_rep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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

    now = time.time()
    upload_idle = upload_thread is None or not upload_thread.is_alive()
    if UPLOADS_ENABLED and upload_idle and now - last_request_time_s >= UPLOAD_INTERVAL_S:
        last_request_time_s = now

        upload_thread = threading.Thread(
            target=send_data_to_thingspeak,
            args=("field1", str(total_rep_count)),
            daemon=True
        )
        upload_thread.start()


if upload_thread is not None:
    upload_thread.join(timeout=5)

pose.close()
cv2.destroyAllWindows()
