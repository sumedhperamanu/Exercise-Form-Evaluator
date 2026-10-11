import mediapipe as mp
import numpy as np
import cv2
from camera import frameCapture

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculateAngle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2

    dot_prod = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    cosine_angle = dot_prod / (mag_v1 * mag_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

for frame in frameCapture():
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].z])
        
        knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z])
        
        # You can use LEFT_ANKLE or LEFT_HEEL depending on your preference
        ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z])

        knee_angle = calculateAngle(hip, knee, ankle)

        status_text = f"Knee Angle: {int(knee_angle)} degrees"
        if knee_angle <= 100:
            depth_text = "Depth Achieved (90 degrees)"
            color = (0, 255, 0)
        else:
            depth_text = "Go deeper"
            color = (0, 0, 255)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("MediaPipe Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pose.close()