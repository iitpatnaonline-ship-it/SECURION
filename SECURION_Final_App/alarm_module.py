from playsound import playsound
import threading
import os

alarm_thread = None
alarm_running = False

def _play_alarm():
    global alarm_running
    alarm_path = os.path.join(os.path.dirname(__file__), "alarm.wav")
    while alarm_running:
        playsound(alarm_path)

def start_alarm():
    global alarm_thread, alarm_running
    if not alarm_running:
        alarm_running = True
        alarm_thread = threading.Thread(target=_play_alarm, daemon=True)
        alarm_thread.start()
        print("🔔 Alarm started")

def stop_alarm():
    global alarm_running
    if alarm_running:
        alarm_running = False
        print("🔕 Alarm stopped")
