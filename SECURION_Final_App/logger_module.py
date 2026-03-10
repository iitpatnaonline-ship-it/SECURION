import csv
import os
import datetime

# Log file ka naam (Excel me open ho jayegi)
LOG_FILE = "security_logs.csv"

def write_log(face_name, risk_level, score, reasons):
    """
    Rishikesh's Logging Module (Day 5 - Day 8)
    Yeh function system ke har event ko CSV file me save karega.
    """
    file_exists = os.path.isfile(LOG_FILE)
    
    # Current time nikalna (Day 7)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Reasons list ko ek single text me badalna
    reason_text = " | ".join(reasons)
    
    try:
        # File ko 'append' (a) mode me open karna taaki purana data delete na ho
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Agar file nayi hai, to pehle headings (Columns) likhein
            if not file_exists:
                writer.writerow(["Timestamp", "Detected Face", "Risk Level", "Score", "Reasons"])
            
            # Data save karein
            writer.writerow([current_time, face_name, risk_level, score, reason_text])
            print(f"📝 [LOG SAVED] {face_name} - {risk_level} (Score: {score})")
            
    except Exception as e:
        print(f"[!] Log save karne me error: {e}")