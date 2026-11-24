import cv2

# Opens camera and hands over frames to main.py

def frameCapture():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not access camera")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        yield frame

    camera.release()
    cv2.destroyAllWindows()
