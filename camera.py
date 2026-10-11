import cv2


def frameCapture():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not access camera")
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        yield frame

    camera.release()
    cv2.destroyAllWindows()
