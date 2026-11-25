import mediapipe as mp
import cv2
import time
from camera import frameCapture
from predict_model import predict_rep
from collections import deque
import requests
from dotenv import load_dotenv
import os
import threading

load_dotenv()

# Initializing variables and configurations

BASE_URL = "https://api.thingspeak.com/update"
THINGSPEAK_API_KEY = os.getenv("THINGSPEAK_API_KEY")  # Read the API key from an ignore .env file for security

# Only allow uploads to ThingSpeak if there is a valid API key
UPLOADS_ENABLED = bool(THINGSPEAK_API_KEY)
if not UPLOADS_ENABLED:
    print("Warning: THINGSPEAK_API_KEY not set, uploads disabled for this run.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define's MediaPipe pose-detection parameters
start_thresh = 0.02
reversal = 0.01
end_thresh = 0.012
calibration_delay = 3
required_landmarks = [11, 12, 23, 24]   # shoulders and hips must be visible to calibrate

feedback = {
    "good": "Good rep",
    "knee_valgus": "Don't let your knees cave",
    "shallow": "Go deeper",
    "shallow + knee_valgus": "Go deeper and don't let your knees cave"
}

standing_hip_y = None
torso_len = None
recent_hips = deque(maxlen=20)
recent_torso = deque(maxlen=20)
rep_start = 0

start_time = time.time()
state = "standing"
current_rep = []
max_displacement = 0
last_relative_depth = 0
relative_depth_samples = []

total_rep_count = 0
rep_counts = {
    "good": 0,
    "shallow": 0,
    "knee_valgus": 0,
    "shallow + knee_valgus": 0
}

message = ""
color = (255, 255, 255)

UPLOAD_INTERVAL_S = 20
last_request_time_s = time.time()

upload_thread = None

# Sending data to Thingspeak

def send_data_to_thingspeak(data_to_send):
    payload = {
        "api_key": THINGSPEAK_API_KEY,
        **data_to_send
    }

    try:
        # Send data via HTTP POST
        response = requests.post(BASE_URL, data=payload, timeout=5)

        # A status code of 200 and an entry ID > 0 means success
        if response.status_code == 200 and int(response.text) > 0:
            print(f"Success! Entry ID: {response.text} | Value: {data_to_send}")
            return True
        else:
            print(f"Failed to update. Response: {response.text}")

    except Exception as e:
        print(f"Error connecting to ThingSpeak: {e}")

    return False

# Set the standing hip height and torso length from the last ~1 second

def calibrate():
    global standing_hip_y, torso_len
    standing_hip_y = sum(recent_hips) / len(recent_hips)
    torso_len = sum(recent_torso) / len(recent_torso)

# State tracker (where in rep user is -- ascending, descending, or standing)
# along with corresponding messages. Tracks landmark data from real-time user squats
# instead of through a csv file, and gives that data directly to predict_model.py to give instant form feedback

for frame in frameCapture():
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Loop through every landmark and, if the visibility is high enough, add it to an array to average hip height and torso length
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
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2

        recent_hips.append(hip_y)
        recent_torso.append(abs(shoulder_y - hip_y))

        visible = all(landmarks[i].visibility > 0.7 for i in required_landmarks)
        still = len(recent_hips) == 20 and max(recent_hips) - min(recent_hips) < 0.006

        if standing_hip_y is None:
            if time.time() - start_time > calibration_delay and still and visible:
                calibrate()
        else:
            displacement = hip_y - standing_hip_y

            if state in ("descending", "ascending") and still and displacement < 0.05:
                if time.perf_counter() - rep_start > 2.0 and visible:
                    calibrate()
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

                # Checks if the user has just changed from descending to ascending
                if displacement < max_displacement - reversal:
                    state = "ascending"
            elif state == "ascending":
                current_rep.append(frame_data)

                # Checks if the user has just finished a rep, and if so, classifies its form using a neural network
                if displacement < end_thresh:
                    if len(current_rep) >= 7:
                        total_rep_count += 1

                        if torso_len:
                            last_relative_depth = max_displacement / torso_len
                            relative_depth_samples.append(last_relative_depth)

                        label = predict_rep(current_rep)['label']
                        message = feedback.get(label, label)
                        color = (0, 255, 0) if label == "good" else (0, 0, 255)

                        if label in rep_counts:
                            rep_counts[label] += 1
                        else:
                            print(f"Warning: Unknown prediction label: {label}")

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
        cv2.putText(frame, f"Reps: {total_rep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Squat Form Evaluator", frame)

    # Program keybinds

    key = cv2.waitKey(1) & 0xFF
    # If ths user presses q: end the program
    if key == ord("q"):
        break
    # If the user presses r: reset all trhe variables for reconfiguration (predominantly used for testing)
    if key == ord("r"):
        standing_hip_y = None
        torso_len = None
        recent_hips.clear()
        recent_torso.clear()
        max_displacement = 0
        last_relative_depth = 0
        relative_depth_samples = []
        start_time = time.time()
        state = "standing"
        current_rep = []
        message = ""

        total_rep_count = 0
        rep_counts = {
            "good": 0,
            "shallow": 0,
            "knee_valgus": 0,
            "shallow + knee_valgus": 0
        }

    # If it has been long enough since the last ThingSpeak upload, and there is valid data to be uploaded, attempt to upload data
    now = time.time()
    upload_idle = upload_thread is None or not upload_thread.is_alive()
    if UPLOADS_ENABLED and upload_idle and now - last_request_time_s >= UPLOAD_INTERVAL_S and total_rep_count > 0:
        last_request_time_s = now

        if len(relative_depth_samples) > 0:
            average_relative_depth = sum(relative_depth_samples) / len(relative_depth_samples)
        else:
            average_relative_depth = 0

        # Data sent over to Thingspeak

        thingspeak_data = {
            "field1": total_rep_count,
            "field2": rep_counts["good"],
            "field3": rep_counts["shallow"],
            "field4": rep_counts["knee_valgus"],
            "field5": rep_counts["shallow + knee_valgus"],
            "field6": round(rep_counts["good"] / total_rep_count, 3),
            "field7": round(average_relative_depth, 3),
            "field8": round(last_relative_depth, 3)
        }

        # Use a thread to allow the camera and neural network to continue capturing new data while current data uploads
        upload_thread = threading.Thread(
            target=send_data_to_thingspeak,
            args=(thingspeak_data,),
            daemon=True
        )
        upload_thread.start()


# If the program is ended, the thread continues trying to upload for 5 more seconds before exiting
if upload_thread is not None:
    upload_thread.join(timeout=5)

pose.close()
cv2.destroyAllWindows()
