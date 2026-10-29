import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime

exercise = "squat"

label = "good"
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

rep_count = 0

standing_hip_y = None
state = "standing"

print("Controls: ")
print("Press space to start/stop recording")
print("Press q to quit")

def save_rep(frames, rep_number):
    if len(frames) < 10:
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

            if standing_hip_y is None:
                standing_hip_y = hip_y

            displacement = hip_y - standing_hip_y

            if state == "standing":
                if displacement > 0.08:
                    state = "descending"
                    current_rep = []
                    print("Squat started")
            elif state == "descending":
                current_rep.append(frame_data)
                if displacement < 0.06:
                    state = "ascending"
            elif state == "ascending":
                current_rep.append(frame_data)
                if displacement < 0.04:

                    if save_rep(current_rep, rep_count + 1):
                        rep_count += 1

                    current_rep = []

                    state = "standing"
                    standing_hip_y = hip_y
                    print(f"Completed squat #{rep_count}")

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
            current_rep = []
            rep_count = 0
            standing_hip_y = None
            state = "standing"
            print("Recording started")
        else:
            recording = False
            print("Recording stopped")
            print(f"Total Reps Recorded: {rep_count}")

            current_rep = []
            standing_hip_y = None
            state = "standing"

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
pose.close()

print("END")
