import cv2
import face_recognition
import pickle
import time
import threading
import os
import numpy as np
from playsound import playsound

# ---------------- SETTINGS ---------------- #
ALARM_DURATION = 30
RESET_HOURS = 5
SAVE_LIMIT = 2
TOLERANCE = 0.55
DETECTION_INTERVAL = 8
TRACKER_TIMEOUT = 2  # seconds without detection refresh
# ------------------------------------------ #

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

if not os.path.exists("captures"):
    os.makedirs("captures")

cap = cv2.VideoCapture(0)

trackers = []
person_memory = {}
frame_count = 0


def play_alarm():
    start = time.time()
    while time.time() - start < ALARM_DURATION:
        playsound("alarm.mp3")


print("Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Lighting normalization (improves recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    current_time = time.time()

    # ---------------- UPDATE TRACKERS ---------------- #
    updated_trackers = []

    for t in trackers:
        success, box = t["tracker"].update(frame)

        if success:
            t["box"] = box

            # remove tracker if not refreshed recently
            if current_time - t["last_seen"] < TRACKER_TIMEOUT:
                updated_trackers.append(t)

            (x, y, w, h) = [int(v) for v in box]
            color = (0, 255, 0) if t["name"] != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, t["name"], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    trackers = updated_trackers

    # ---------------- FACE DETECTION ---------------- #
    if frame_count % DETECTION_INTERVAL == 0:

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):

            face_distances = face_recognition.face_distance(known_encodings, encoding)

            name = "Unknown"

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] < TOLERANCE:
                    name = known_names[best_match_index]

            new_box = (left, top, right - left, bottom - top)

            matched_tracker = None

            for t in trackers:
                (x, y, w, h) = [int(v) for v in t["box"]]

                if abs(x - left) < 50 and abs(y - top) < 50:
                    matched_tracker = t
                    break

            if matched_tracker:
                matched_tracker["last_seen"] = current_time
                continue

            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, new_box)

            trackers.append({
                "tracker": tracker,
                "name": name,
                "box": new_box,
                "last_seen": current_time
            })

            # ---------------- ALARM & SAVE ---------------- #
            if name != "Unknown":

                if name not in person_memory:
                    person_memory[name] = {
                        "last_reset": 0,
                        "captures": 0,
                        "alarm_played": False
                    }

                pdata = person_memory[name]

                if current_time - pdata["last_reset"] > RESET_HOURS * 3600:
                    pdata["last_reset"] = current_time
                    pdata["captures"] = 0
                    pdata["alarm_played"] = False

                if not pdata["alarm_played"]:
                    threading.Thread(target=play_alarm, daemon=True).start()
                    pdata["alarm_played"] = True

                if pdata["captures"] < SAVE_LIMIT:
                    filename = f"captures/{name}_{int(current_time)}.jpg"
                    cv2.imwrite(filename, frame)
                    pdata["captures"] += 1

    cv2.imshow("Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
