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



class DriverMonitorApp:
    def __init__(self):
        
        # Initialize the video capture
        self.cap = cv2.VideoCapture("http://127.0.0.1:5000/video_feed2")
        

        if not self.cap.isOpened():
            print("⚠️ Error: Could not open the camera!")
            sys.exit(1)
        else:
            print("✅ Camera successfully opened!")

        self.frame = None  # Add this in __init__()
        self.running = True  # Add a flag to stop threads properly

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
        Updates the GUI labels based on the distraction status (head and gaze).
        Runs continuously every 500ms to refresh the displayed information.
        """

        # Define colors
        dark_orange = "#FF8C00"  # Warning color
        grey = "#808080"  # Default color for baseline setting
        warning_text = ""  # Initialize empty warning message
        warning_color = "black"  # Default text color

        # # Call the JSON update for camera2:
        self.update_status_file_camera2()


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
    
    
    def update_status_file_camera2(self):
        # Collect the current status values from the Tkinter labels.
        status_camera2 = {
            "gaze_center": gh.gaze_gui,  # Assuming this is a string like "Center: (x, y)"
            "gaze_status": gh.gaze_status_gui,  # e.g., "Status: NORMAL ✅"
            "pitch": f"Pitch: {gh.pitch_gui:.2f} deg",
            "yaw": f"Yaw: {gh.yaw_gui:.2f} deg",
            "roll": f"Roll: {gh.roll_gui:.2f} deg",
            "head_status": gh.head_status_gui,  # e.g., "Head Status: NORMAL"
            "distraction": f"Distraction Count within 3 min: {gh.distraction_counter}",
            "blinks": f"num of blinks: {yb.num_of_blinks_gui}",
            "microsleep_duration": f"microsleep duration: {yb.microsleep_duration_gui:.2f} sec",
            "yawns": f"num of yawns: {yb.num_of_yawns_gui}",
            "yawn_duration": f"yawn duration: {yb.yawn_duration_gui:.2f} sec",
            "blinks_per_minute": f"blinks/min: {yb.blinks_per_minute_gui}",
            "yawns_per_minute": f"yawns/min: {yb.yawns_per_minute_gui}",
            "alert": gh.flag_gui  # or any custom fatigue alert string you assign here
        }
        
        try:
            with open("status_driver_fatigue.json", "w") as f:
                json.dump(status_camera2, f)
        except Exception as e:
            print("Error writing status.json:", e)
            

if __name__ == "__main__":
    app = DriverMonitorApp()
    try:
        app.run()  # This will call update_info repeatedly in a loop
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        app.close_app()


 