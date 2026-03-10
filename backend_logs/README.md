# SECURION – AI Powered Surveillance System (Backend Module)

## Project Overview
SECURION एक AI based security monitoring system है।

यह system camera detection events और face recognition results को backend में log करता है।

यह logs security monitoring, risk detection और alert system के लिए उपयोगी होते हैं।

---

## Backend Development Timeline

### Week 2 – Core Logging System

**Day 12 – Camera Event Logging**  
Camera detection events CSV log file में store किए गए।

**Day 13 – Face Match Logging**  
AI face recognition result backend में log किया गया।

**Day 14 – Context Logging**  
Logs में LocalTime, UTCTime और Timezone add किया गया।

**Day 15 – Full System Log Test**  
Camera + AI + Risk logic integrate करके complete log बनाया गया।

**Day 16 – Duplicate Log Avoid System**  
Same event बार-बार log होने से रोकने के लिए duplicate check add किया गया।

**Day 17 – Error Handling Backend**  
File missing या permission error होने पर backend crash न हो इसलिए error handling add की।

**Day 18 – Log Size Handling**  
Log file size limit और log rotation system implement किया गया।

**Day 19 – Human Readable Logs**  
Logs CSV के साथ-साथ readable text format में भी store किए गए।

**Day 20 – GitHub Integration**  
Backend project को organized structure में GitHub पर upload किया गया।

---

### Week 5 – Report & Dashboard Support

**Day 21 – Alert Count Summary**  
Log file में alert events count किए गए।

**Day 22 – Statistics Generation**  
Total alerts, high risk और suspicious events calculate किए गए।

**Day 23 – Backend Explanation**  
Backend working documentation और report explanation तैयार किया गया।

**Day 24 – CSV to Table View**  
CSV logs को table format में display किया गया।

---

## Log Structure

Backend system events को CSV format में store करता है।

Log file fields:

- Time
- CameraID
- PersonID
- MatchScore
- MatchResult
- Risk
- Event

Example log entry:
2026-03-09 10:29:54 | cam_01 | om1 | 0.75 | Matched | Low | Detected

---

## Risk Decision Logic

Backend risk level decide करता है based on recognition result:

- Unknown person → **High Risk (Alert)**
- Low match score → **Medium Risk (Suspicious)**
- Matched face → **Low Risk (Detected)**

---

## Backend Role in SECURION

Backend system surveillance events का complete record maintain करता है।

यह system activity history store करता है जिससे security analysis, alerts और reports generate किए जा सकते हैं।

---

## Tools Used

Backend development में उपयोग किए गए tools:

- Python
- CSV Module
- Datetime Module
- Visual Studio Code
- GitHub
- Excel (log analysis)
---

## Backend Role in SECURION

The backend module handles event logging, risk detection and security monitoring.

It collects camera detection events and face recognition results and stores them in structured log files.

These logs help in:

- Security monitoring
- Risk detection
- Alert generation
- Data analysis

Backend also ensures reliable logging with error handling and duplicate log prevention.
