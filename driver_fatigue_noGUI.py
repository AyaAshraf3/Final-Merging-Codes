import threading
import cv2
import time
import sys
from thresholds import *
from PIL import Image, ImageTk  
import importlib

from yawnBlink_farah import DrowsinessDetector
from gazeHead_farah import GazeAndHeadDetection  # Import your class
import os
import json
from flask import Flask, Response



import yawnBlink_farah as yb  # Import the script to access its global variables
importlib.reload(yb)  # ✅ This forces Python to reload the latest changes

import gazeHead_farah as gh  # Import the script to access its global variables
importlib.reload(gh)  # ✅ This forces Python to reload the latest changes


JSON_PATH = "driver_assistant.json"

class DriverMonitorApp:
    def __init__(self):
        # Initialize alert variables
        self.Distraction_Alert = "Driver is awake and focused"
        self.Fatigue_Alert = "Driver is awake and focused"
        self.sleep_alert = "Driver is awake and focused"
        
        # Initialize the video capture
        self.cap = cv2.VideoCapture("http://127.0.0.1:5000/video_feed2")
        
        if not self.cap.isOpened():
            print("⚠️ Error: Could not open the camera!")
            sys.exit(1)
        else:
            print("✅ Camera successfully opened!")

        self.frame = None
        self.running = True

        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()

        # Create instances of both detection classes
        self.gaze_detector = GazeAndHeadDetection()
        self.yawn_blink_detector = DrowsinessDetector()

        # Start processing in separate threads
        self.gaze_thread = threading.Thread(target=self.process_gaze_head, daemon=True)
        self.yawn_thread = threading.Thread(target=self.process_yawn_blink, daemon=True)

        self.gaze_thread.start()
        self.yawn_thread.start()

        # Start GUI update loop
        self.update_info()

    def process_gaze_head(self):
        """ Runs gaze and head movement detection in a separate thread """
        while self.running:
            if self.frame is None:
                continue
            self.gaze_detector.process_frame(self.frame.copy())

    def process_yawn_blink(self):
        """ Runs yawn and blink detection in a separate thread """
        while self.running:
            if self.frame is None:
                continue
            self.yawn_blink_detector.process_frames(self.frame.copy())


    def update_info(self):
        """
        Updates the status information and alerts based on the distraction and fatigue status.
        Runs continuously every 500ms to refresh the displayed information.
        """
        # Get current values
        up_pitch_gui = gh.pitch_gui 
        up_yaw_gui = gh.yaw_gui 
        up_roll_gui = gh.roll_gui 
        up_gaze_gui = gh.gaze_gui 
        up_gaze_status_gui = gh.gaze_status_gui 
        up_head_status_gui = gh.head_status_gui 
        up_flag_gui = gh.flag_gui
        up_distraction_flag_head = gh.distraction_flag_head 
        up_distraction_flag_gaze = gh.distraction_flag_gaze 
        up_temp = gh.temp 
        up_temp_g = gh.temp_g 
        up_distraction_counter = gh.distraction_counter 
        up_gaze_flag = gh.gaze_flag
        up_buzzer_running = gh.buzzer_running 

        up_num_of_blinks_gui = yb.num_of_blinks_gui 
        up_microsleep_duration_gui = yb.microsleep_duration_gui 
        up_num_of_yawns_gui = yb.num_of_yawns_gui 
        up_yawn_duration_gui = yb.yawn_duration_gui 
        up_blinks_per_minute_gui = yb.blinks_per_minute_gui 
        up_yawns_per_minute_gui = yb.yawns_per_minute_gui

        # Initialize alert text and color
        alert_text = "Driver is awake and focused"
        alert_color = "black"

        # Reset alerts to default state
        self.Distraction_Alert = "off"
        self.Fatigue_Alert = "off"
        self.sleep_alert = "off"

        # Check for distraction warnings
        if up_flag_gui == 1:  # Only check when baseline is set
            # Gaze Warnings
            if up_distraction_flag_gaze == 2 and up_temp_g == 1:  # High risk due to gaze
                alert_text = "🚨 HIGH RISK: Driver Distracted!"
                self.Distraction_Alert = "on"
            elif up_distraction_flag_gaze == 1 and up_temp_g == 0:  # Moderate distraction due to gaze
                alert_text = "⚠️WARNING: Driver Distracted!"
                self.Distraction_Alert = "on"

            # Head Movement Warnings
            if up_distraction_flag_head == 2 and up_temp == 1:  # High risk due to head movement
                alert_text = "🚨 HIGH RISK: Driver Distracted!"
                self.Distraction_Alert = "on"
            elif up_distraction_flag_head == 1 and up_temp == 0:  # Moderate distraction due to head movement
                alert_text = "⚠️WARNING: Driver Distracted!"
                self.Distraction_Alert = "on"

        # Check for sleep warnings
        if round(up_microsleep_duration_gui, 2) > microsleep_threshold:
            alert_text = "⚠️Alert: Prolonged Microsleep Detected!"
            self.sleep_alert = "on"
        elif up_gaze_gui == "Down" and up_gaze_status_gui == "ABNORMAL GAZE" and up_head_status_gui == "ABNORMAL PITCH":
            alert_text = "⚠️Alert! Driver is fainted :("
            self.sleep_alert = "on"
        # Check for fatigue warnings
        elif round(up_yawn_duration_gui, 2) > yawning_threshold:
            alert_text = "⚠️Alert: Prolonged Yawn Detected!"
            self.Fatigue_Alert = "on"
        elif up_microsleep_duration_gui > microsleep_threshold:
            alert_text = "⚠️Alert! Possible Fatigue!"
            self.Fatigue_Alert = "on"
        elif up_blinks_per_minute_gui > 35 or up_yawns_per_minute_gui > 5:
            alert_text = "⚠️Alert! Driver is Highly Fatigued!"
            self.Fatigue_Alert = "on"
        elif up_blinks_per_minute_gui > 25 or up_yawns_per_minute_gui > 3:
            alert_text = "⚠️ Alert! Driver is Possibly Fatigued!"
            self.Fatigue_Alert = "on"

        # Update both JSON files
        self.update_status_file_camera2(alert_text)
        self.update_driver_assistant_field(
            Fatigue_Alert=self.Fatigue_Alert,
            Distraction_Alert=self.Distraction_Alert,
            sleep_alert=self.sleep_alert
        )
        print(f"Updated alerts - Fatigue: {self.Fatigue_Alert}, Distraction: {self.Distraction_Alert}, Sleep: {self.sleep_alert}")

    def run(self):
        try:
            while self.running:
                self.update_info()
                time.sleep(0.5)  # 500ms
        except KeyboardInterrupt:
            print("Interrupted. Shutting down.")
            self.close_app()

    def close_app(self):
        self.cap.release()
        self.running = False
        print("App closed cleanly.")


    def capture_frames(self):
        """ Continuously captures frames from the camera """
        while self.running:
            ret, frame = self.cap.read()
            #frame = self.cap.get_frame()  # Use the shared camera's getter
            if ret:
            #if frame is not None:
                self.frame = frame
                print("Frame Captured!")  # ✅ Debugging
            else:
                print("Failed to capture frame!")  # ✅ Debugging
    
    
    def update_status_file_camera2(self, alert_text=""):
        # Collect the current status values
        status_camera2 = {
            "gaze_center": gh.gaze_gui,
            "gaze_status": gh.gaze_status_gui,
            "pitch": f"Pitch: {gh.pitch_gui:.2f} deg",
            "yaw": f"Yaw: {gh.yaw_gui:.2f} deg",
            "roll": f"Roll: {gh.roll_gui:.2f} deg",
            "head_status": gh.head_status_gui,
            "distraction": f"Distraction Count within 3 min: {gh.distraction_counter}",
            "blinks": f"num of blinks: {yb.num_of_blinks_gui}",
            "microsleep_duration": f"microsleep duration: {yb.microsleep_duration_gui:.2f} sec",
            "yawns": f"num of yawns: {yb.num_of_yawns_gui}",
            "yawn_duration": f"yawn duration: {yb.yawn_duration_gui:.2f} sec",
            "blinks_per_minute": f"blinks/min: {yb.blinks_per_minute_gui}",
            "yawns_per_minute": f"yawns/min: {yb.yawns_per_minute_gui}",
            "alert": alert_text
        }
        
        try:
            with open("status_driver_fatigue.json", "w") as f:
                json.dump(status_camera2, f, indent=2)
            print("Successfully updated status_driver_fatigue.json")
        except Exception as e:
            print(f"Error writing status_driver_fatigue.json: {e}")

    
    def update_driver_assistant_field(self,**field_updates):
        """
        Update multiple fields in the driver assistant JSON file simultaneously.
        Args:
            **field_updates: Keyword arguments where key is field_name and value is new_value
        """
        try:
            # 1. Read existing data (or start fresh)
            if os.path.exists(JSON_PATH):
                with open(JSON_PATH, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            # 2. Update all fields at once
            data.update(field_updates)

            # 3. Write directly to the file
            with open(JSON_PATH, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error updating driver assistant fields: {e}")
            

if __name__ == "__main__":
    app = DriverMonitorApp()
    try:
        app.run()  # This will call update_info repeatedly in a loop
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        app.close_app()


 