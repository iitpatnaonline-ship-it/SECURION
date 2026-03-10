import face_recognition
import cv2
import os
import numpy as np

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    print(f"[*] Scanning directory: {known_faces_dir}")
    
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(known_faces_dir, filename)
            print(f"[*] Processing: {filename}")
            
            try:
                # 1. OpenCV से इमेज पढ़ें
                img = cv2.imread(path)
                if img is None:
                    print(f"[!] Error: Could not read {filename}")
                    continue
                    
                # 2. अगर इमेज बहुत बड़ी है, तो उसे छोटा करें (ताकि dlib क्रैश ना हो)
                height, width = img.shape[:2]
                max_size = 800
                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                
                # 3. BGR से RGB में बदलें
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 4. BULLETPROOF STEP: ज़बरदस्ती एक नया और क्लीन 8-bit array बनाएं
                clean_rgb = np.ascontiguousarray(img_rgb[:, :, :3], dtype=np.uint8)
                
                print(f"    -> Shape: {clean_rgb.shape}, Type: {clean_rgb.dtype}")

                # 5. चेहरे को एन्कोड करें
                encodings = face_recognition.face_encodings(clean_rgb)
                
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(filename)[0])
                    print(f"[+] Successfully loaded: {filename}")
                else:
                    print(f"[-] No face found in: {filename}")
            
            except Exception as e:
                print(f"[!] Error processing {filename}: {e}")
                
    return known_encodings, known_names

def match_face(frame, known_encodings, known_names):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    clean_rgb = np.ascontiguousarray(img_rgb[:, :, :3], dtype=np.uint8)
    
    face_locations = face_recognition.face_locations(clean_rgb)
    face_encodings = face_recognition.face_encodings(clean_rgb, face_locations)
    
    results = []
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        
        if known_encodings:
            # 1. सभी known चेहरों से 'दूरी' (distance) निकालें
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # 2. जो चेहरा सबसे ज़्यादा मैच कर रहा है (जिसकी दूरी सबसे कम है), उसका इंडेक्स निकालें
            best_match_index = np.argmin(face_distances)
            
            # 3. Tolerance को 0.5 पर सेट करें (ताकि AI थोड़ा सख्त हो जाए और कंफ्यूज ना हो)
            if face_distances[best_match_index] < 0.5:
                name = known_names[best_match_index]
                
        results.append((name, (top, right, bottom, left)))
        
    return results