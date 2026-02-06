import cv2
import os
import time

from camera_module import start_camera, read_frame, stop_camera


# Load Haarcascade
cascade_path = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("❌ Haarcascade XML not loaded")
    exit()

# Start camera
cap = start_camera(fps=20)

prev_time = 0

while True:
    frame = read_frame(cap)
    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=7,
    minSize=(60, 60)


    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("SECURION - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# FPS manual cap (20 FPS)
time.sleep(1 / 20)

stop_camera(cap)
