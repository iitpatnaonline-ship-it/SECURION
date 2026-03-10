"""
=============================================================================
PROJECT SECURION - AI Powered Context-Aware Surveillance System
=============================================================================
Module: Main Integration Engine (Smart Camera)
Lead Developer: Shivam (Project Lead)
Day 26-28: Final Testing & UI Cleanup
Description: Clean UI, removed FPS and Score clutter for professional demo.
=============================================================================
"""

import cv2
import time

# --- MODULE IMPORTS ---
try:
    from face_matcher import load_known_faces, match_face
    from alarm_module import start_alarm, stop_alarm
    from context_engine import calculate_risk_score
    from logger_module import write_log  
except ImportError as e:
    print(f"[CRITICAL ERROR] Modules missing or corrupted: {e}")
    exit()

# --- SYSTEM SETTINGS ---
COOLDOWN_TIME = 5        
ZONE_TYPE = "Normal"     

print("\n[*] Initializing SECURION System...")
known_encodings, known_names = load_known_faces("known_faces")

if not known_encodings:
    print("[!] Warning: Face database ('known_faces') is empty.")
    exit()

# --- AUTO-DETECT CAMERA LOGIC ---
def auto_detect_camera():
    print("[*] Scanning for available cameras...")
    for index in range(3):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[+] Local Camera detected at Index: {index}")
                return cap
            
    print("[!] No local camera. Attempting DroidCam connection...")
    droidcam_ip = "http://192.168.43.1:4747/video"
    cap = cv2.VideoCapture(droidcam_ip)
    if cap.isOpened():
        print("[+] DroidCam connected successfully.")
        return cap

    return None

cap = auto_detect_camera()

if cap is None:
    print("[CRITICAL ERROR] No video source found. Shutting down.")
    exit()

print("[*] System Armed and Running. Press 'Q' to terminate.\n")

# --- GLOBAL VARIABLES ---
alarm_on = False
last_alarm_time = 0
last_log_time = 0  

# --- MAIN SURVEILLANCE LOOP ---
while True:
    try:
        ret, frame = cap.read()
        
        if not ret:
            print("[!] Frame drop detected. Attempting recovery...")
            time.sleep(2)
            continue  

        # --- AI FACE DETECTION ---
        results = match_face(frame, known_encodings, known_names)
        current_time = time.time()
        
        highest_risk_score = 0
        display_status = "SAFE"
        system_color = (0, 255, 0)
        trigger_alarm = False
        
        target_name = "None"
        target_risk_level = "LOW"
        target_reasons = []

        # --- CONTEXT & RISK ANALYSIS ---
        for name, (top, right, bottom, left) in results:
            face_status = "Unknown" if name == "Unknown" else "Family"

            # Calculate risk via Gyanesh's Context Engine
            score, risk_level, reasons = calculate_risk_score(face_status, ZONE_TYPE)
            
            if score > highest_risk_score:
                highest_risk_score = score
                display_status = risk_level
                target_name = name
                target_risk_level = risk_level
                target_reasons = reasons

            # Assign Bounding Box Colors Based on Risk
            if score >= 70:
                box_color = (0, 0, 255)     # Red (High Risk)
            elif score >= 40:
                box_color = (0, 255, 255)   # Yellow (Medium Risk)
            else:
                box_color = (0, 255, 0)     # Green (Safe)
            
            # ---------------------------------------------------------
            # CLEAN UI UPDATE: Draw Box and Name only (No Score shown)
            # ---------------------------------------------------------
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame, f"{name}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
        # Determine Global System State
        if highest_risk_score >= 70:
            system_color = (0, 0, 255) 
            trigger_alarm = True
        elif highest_risk_score >= 40:
            system_color = (0, 255, 255) 

        # --- LOGGING (Rishikesh's Module) ---
        if highest_risk_score >= 40:
            if (current_time - last_log_time) >= COOLDOWN_TIME:
                write_log(target_name, target_risk_level, highest_risk_score, target_reasons)
                last_log_time = current_time  

        # --- ALARM TRIGGER ---
        if trigger_alarm:
            if (not alarm_on) and (current_time - last_alarm_time >= COOLDOWN_TIME):
                start_alarm()
                alarm_on = True
                last_alarm_time = current_time
                print(f"🚨 [ALERT] High Risk Event Detected - Alarm Triggered!")
        else:
            if alarm_on:
                stop_alarm()
                alarm_on = False
                print("🔕 [SAFE] Risk normalized - Alarm Stopped.")

        # ---------------------------------------------------------
        # CLEAN UI UPDATE: Display Status only (No FPS clutter)
        # ---------------------------------------------------------
        cv2.putText(frame, f"STATUS: {display_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, system_color, 2)
        
        # Render Resizable Window
        cv2.namedWindow("SECURION - Core Engine", cv2.WINDOW_NORMAL)
        cv2.imshow("SECURION - Core Engine", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[*] Manual shutdown initiated by user.")
            break

    except Exception as e:
        print(f"[!] System Warning - Non-fatal error: {e}")
        time.sleep(1)

# --- SAFE SHUTDOWN SEQUENCE ---
if alarm_on:
    stop_alarm()
cap.release()
cv2.destroyAllWindows()
print("[*] SECURION System safely terminated.")