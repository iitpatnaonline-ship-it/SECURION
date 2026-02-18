import face_recognition
import cv2
import os
import numpy as np

# ---------- LOAD KNOWN FACES ----------
known_encodings = []
known_names = []

for file in os.listdir("known_faces"):
    img = face_recognition.load_image_file(f"known_faces/{file}")
    enc = face_recognition.face_encodings(img)[0]
    known_encodings.append(enc)
    known_names.append(os.path.splitext(file)[0])

print("Known faces loaded:", known_names)

# ---------- START CAMERA ----------
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    # resize frame for faster processing
    small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # detect faces
    locations = face_recognition.face_locations(rgb_small, model="hog")
    encodings = face_recognition.face_encodings(rgb_small, locations)

    # ---------- LOOP THROUGH ALL FACES ----------
    for (top, right, bottom, left), face_encoding in zip(locations, encodings):

        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        color = (0,0,255)

        if len(distances) > 0:
            best_match = np.argmin(distances)
            if matches[best_match]:
                name = known_names[best_match]
                color = (0,255,0)

        # scale box back to original size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # draw rectangle
        cv2.rectangle(frame,(left,top),(right,bottom),color,2)

        # label
        cv2.putText(frame,name,(left,top-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    # show window
    cv2.imshow("Face Recognition", frame)

    # press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
