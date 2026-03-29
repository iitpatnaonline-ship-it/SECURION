import cv2
import face_recognition
import pickle
import numpy as np
import time
import os
import threading
from playsound import playsound

# ---------------- SETTINGS ---------------- #
TOLERANCE = 0.55
ALARM_DURATION = 30
SAVE_INTERVAL_HOURS = 8
DETECTION_SCALE = 0.5
FRAME_SKIP = 3
BUFFER_SIZE = 5
# ------------------------------------------ #

# ---------------- LOAD DATA ---------------- #
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

os.makedirs("captures/known", exist_ok=True)

# ---------------- ALARM ---------------- #
def play_alarm():
    start = time.time()
    while time.time() - start < ALARM_DURATION:
        playsound("alarm.mp3")

# ---------------- WEAPON MODEL ---------------- #
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

weapon_classes = ["knife", "gun"]

output_layers = net.getUnconnectedOutLayersNames()

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0)

person_memory = {}

# 🔥 NEW: smoothing memory
name_buffer = []
frame_count = 0

print("[INFO] Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Lighting improvement
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    frame_count += 1

    # ---------------- FACE RECOGNITION ---------------- #
    if frame_count % FRAME_SKIP == 0:

        small = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, boxes)

        for (top, right, bottom, left), encoding in zip(boxes, encodings):

            matches = face_recognition.compare_faces(
                known_encodings, encoding, tolerance=TOLERANCE)

            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_encodings, encoding)

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    name = known_names[best_match]

            # 🔥 SMOOTHING BUFFER
            name_buffer.append(name)
            if len(name_buffer) > BUFFER_SIZE:
                name_buffer.pop(0)

            final_name = max(set(name_buffer), key=name_buffer.count)

            # scale back
            top = int(top / DETECTION_SCALE)
            right = int(right / DETECTION_SCALE)
            bottom = int(bottom / DETECTION_SCALE)
            left = int(left / DETECTION_SCALE)

            color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, final_name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # ---------------- ONLY KNOWN ACTION ---------------- #
            if final_name != "Unknown":

                if final_name not in person_memory:
                    person_memory[final_name] = {
                        "last_time": 0,
                        "alarm": False
                    }

                pdata = person_memory[final_name]

                if current_time - pdata["last_time"] > SAVE_INTERVAL_HOURS * 3600:
                    pdata["alarm"] = False

                if not pdata["alarm"]:
                    threading.Thread(target=play_alarm, daemon=True).start()
                    pdata["alarm"] = True
                    pdata["last_time"] = current_time

                    filename = f"captures/known/{final_name}_{int(current_time)}.jpg"
                    cv2.imwrite(filename, frame)

    # ---------------- WEAPON DETECTION ---------------- #
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                label = classes[class_id]

                if label in weapon_classes:

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2)
                    cv2.putText(frame, f"WEAPON: {label}",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 0), 2)

                    threading.Thread(target=play_alarm, daemon=True).start()

    cv2.imshow("Face & Weapon Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
