#Camera + FPS + Resize + Control
import cv2
import time

def start_camera(camera_id=0, fps=20):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("❌ Camera not opening")
        return None

    cap.set(cv2.CAP_PROP_FPS, fps)
    print(f"✅ Camera started at {fps} FPS")
    return cap


def read_frame(cap, width=640, height=480):
    ret, frame = cap.read()
    if not ret:
        return None

    # Resize for performance
    frame = cv2.resize(frame, (width, height))
    return frame


def stop_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera closed")
