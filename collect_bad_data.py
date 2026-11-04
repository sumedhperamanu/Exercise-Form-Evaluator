import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime
import time

exercise = "squat"

label = "knee_valgus"
data_dir = f"data/{exercise}/{label}"
os.makedirs(data_dir, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode = False,
    model_complexity = 1,
    enable_segmentation = False,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

camera = cv2.VideoCapture(0)

recording = False
current_rep = []
standing_samples = []

rep_count = 0

standing_hip_y = None
state = "standing"
max_displacement = 0
record_start = 0

start_thresh = 0.02
reversal = 0.01
end_thresh = 0.012
calibration_delay = 3

print("Controls: ")

print("Press space to start/stop recording")
print("Press q to quit")

def save_rep(frames, rep_number):
    if len(frames) < 5:
            print("Rep too short - not saved")
            return False

    file_number = 0

    while True:
        file_name = os.path.join(
            data_dir,
            f"Squat_{file_number:03d}.csv"
        )
        if not os.path.exists(file_name):
            break

        file_number += 1

    recording_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", recording_time])
        writer.writerow([])
        header = []

        for i in range(33):
            header.extend([
                f"x_{i}",
                f"y_{i}",
                f"z_{i}",
                f"visibility_{i}"
            ])

        writer.writerow(header)
        writer.writerows(frames)

    print(f"Saved rep {rep_number}: {file_name}")
    print(f"Frames: {len(frames)}")

    return True


while True:
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        left_hip = 23
        right_hip = 24

        left_hip_y = landmarks[left_hip].y
        right_hip_y = landmarks[right_hip].y

        hip_y = (left_hip_y + right_hip_y) / 2

        if recording:
            frame_data = []

            for landmark in landmarks:
                frame_data.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])

            if standing_hip_y is None and time.time() - record_start > calibration_delay:
                standing_samples.append(hip_y)
                if len(standing_samples) >= 30:
                    standing_hip_y = sum(standing_samples) / len(standing_samples)
                    standing_samples = []

            if standing_hip_y is not None:
                displacement = hip_y - standing_hip_y
                print(f"Displacement: {displacement:.3f}, State: {state}")

                if state == "standing":
                    if displacement > start_thresh:
                        state = "descending"
                        current_rep = []
                        max_displacement = displacement
                        print("Squat started")
                elif state == "descending":
                    current_rep.append(frame_data)
                    max_displacement = max(max_displacement, displacement)
                    if displacement < max_displacement - reversal:
                        state = "ascending"
                elif state == "ascending":
                    current_rep.append(frame_data)
                    if displacement < end_thresh:

                        if save_rep(current_rep, rep_count + 1):
                            rep_count += 1
                            print(f"Completed squat #{rep_count}")

                        current_rep = []
                        state = "standing"

                previous_displacement = displacement

    if recording:
        status = "RECORDING"
    else:
        status = "NOT RECORDING"

    cv2.putText(
        frame,
        f"Exercise {exercise}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Label {label}",
        (20,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Rep Count: {rep_count}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,255) if recording else (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"State: {state}",
        (20,150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Exercise Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        if not recording:
            recording = True
            standing_samples = []
            record_start = time.time()
            current_rep = []
            rep_count = 0
            standing_hip_y = None
            state = "standing"
            max_displacement = 0
            print("Recording started")
        else:
            recording = False
            print("Recording stopped")
            print(f"Total Reps Recorded: {rep_count}")

            current_rep = []
            standing_samples = []
            standing_hip_y = None
            state = "standing"

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
pose.close()

print("END")
