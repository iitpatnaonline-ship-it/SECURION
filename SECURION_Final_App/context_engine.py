import datetime

def calculate_risk_score(face_status, zone_type="Normal"):
    """
    face_status: "Family", "Unknown", ya "Target"
    zone_type: "Normal" ya "Restricted"
    """
    total_score = 0
    reasons = [] # Log file me save karne ke liye karan

    # -----------------------------
    # 1. FACE LOGIC (Shubham's Module Logic)
    # -----------------------------
    if face_status == "Target":
        total_score += 100
        reasons.append("🚨 WANTED TARGET DETECTED (+100)")
    elif face_status == "Unknown":
        total_score += 50
        reasons.append("👤 Unknown Person (+50)")
    elif face_status == "Family":
        total_score += 0
        reasons.append("✅ Known Family/Staff (+0)")

    # -----------------------------
    # 2. TIME LOGIC (Gyanesh's Module)
    # -----------------------------
    #current_hour = datetime.datetime.now().hour
    current_hour = 23
    # Raat ka samay: Raat 10 baje (22) se subah 6 baje (6) tak
    if current_hour >= 22 or current_hour < 6:
        total_score += 20
        reasons.append("🌙 Night Time (+20)")
    else:
        reasons.append("☀️ Day Time (+0)")

    # -----------------------------
    # 3. ZONE LOGIC (Gyanesh's Module)
    # -----------------------------
    if zone_type == "Restricted":
        total_score += 30
        reasons.append("🚫 Restricted Zone (+30)")
    else:
        reasons.append("🟢 Normal Zone (+0)")

    # -----------------------------
    # 4. DECISION (Alert ya Safe?)
    # -----------------------------
    if total_score >= 70:
        risk_level = "HIGH RISK - ALARM"
    elif total_score >= 40:
        risk_level = "MEDIUM RISK - LOG ONLY"
    else:
        risk_level = "LOW RISK - SAFE"

    return total_score, risk_level, reasons

# --- Yahan sirf test karne ke liye chota sa code (Aap ise run karke dekh sakte hain) ---
if __name__ == "__main__":
    print("--- CONTEXT ENGINE TEST ---")
    
    # Test Case: Raat ke samay koi unknown aata hai
    score, level, reason_list = calculate_risk_score(face_status="Unknown", zone_type="Normal")
    
    print(f"Total Score: {score}")
    print(f"Risk Level: {level}")
    print("Reasons:")
    for r in reason_list:
        print(f" - {r}")