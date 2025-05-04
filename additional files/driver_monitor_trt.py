import threading
import cv2
import time
import sys
import tkinter as tk
from thresholds import *
from PIL import Image, ImageTk  
import importlib

from yawnBlink_farah_trt import DrowsinessDetector
from gazeHead_farah_trt import GazeAndHeadDetection
import os
import json

import yawnBlink_farah_trt as yb
importlib.reload(yb)

import gazeHead_farah_trt as gh
importlib.reload(gh)

class DriverMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Monitoring System")
        
        # ========================= TITLE SECTION ========================= #
        self.title_label = tk.Label(
            root, text="Driver's Fatigue and Distraction Module",
            font=("Courier New", 22, "bold"), fg="blue"
        )
        self.title_label.pack(pady=10)

        # ========================= MAIN CONTAINER ========================= #
        frame_container = tk.Frame(root)
        frame_container.pack(fill="both", expand=True, padx=10, pady=10)

        # ========================= VIDEO FEED SECTION (LEFT SIDE) ========================= #
        self.video_label = tk.Label(frame_container)
        self.video_label.grid(row=0, column=0, rowspan=10, padx=10, pady=10)

        # ========================= TEXT OUTPUT SECTION (RIGHT SIDE) ========================= #
        text_frame = tk.Frame(frame_container)
        text_frame.grid(row=0, column=1, padx=30, pady=30, sticky="nw")

        # ========================= GAZE DETECTION SECTION ========================= #
        gaze_frame = tk.Frame(text_frame)
        gaze_frame.grid(row=0, column=0, sticky="w")

        self.gaze_label = tk.Label(gaze_frame, text="GAZE DETECTION", font=("Arial", 11, "bold"))
        self.gaze_label.pack(anchor="w")

        self.gaze_center_label = tk.Label(gaze_frame, text="Center: ", font=("Arial", 11))
        self.gaze_center_label.pack(anchor="w")

        self.gaze_status_label = tk.Label(gaze_frame, text="Status: ", font=("Arial", 11))
        self.gaze_status_label.pack(anchor="w")

        # ========================= HEAD DETECTION SECTION ========================= #
        head_frame = tk.Frame(text_frame)
        head_frame.grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.head_label = tk.Label(head_frame, text="HEAD MOVEMENT", font=("Arial", 11, "bold"))
        self.head_label.pack(anchor="w")

        self.pitch_label = tk.Label(head_frame, text="Pitch: ", font=("Arial", 11))
        self.pitch_label.pack(anchor="w")

        self.yaw_label = tk.Label(head_frame, text="Yaw: ", font=("Arial", 11))
        self.yaw_label.pack(anchor="w")

        self.roll_label = tk.Label(head_frame, text="Roll: ", font=("Arial", 11))
        self.roll_label.pack(anchor="w")

        self.head_status_label = tk.Label(head_frame, text="Status: ", font=("Arial", 11))
        self.head_status_label.pack(anchor="w")

        # ========================= DISTRACTION SECTION ========================= #
        distraction_frame = tk.Frame(text_frame)
        distraction_frame.grid(row=2, column=0, sticky="w")

        self.distraction_label = tk.Label(distraction_frame, text="Num of Distraction / 3 min:", font=("Arial", 11))
        self.distraction_label.pack(anchor="w")

        self.distraction_flag_label = tk.Label(distraction_frame, text="", font=("Arial", 11, "bold"))
        self.distraction_flag_label.pack(anchor="w")

        self.separator2 = tk.Label(distraction_frame, text="--------------------------------------------------", font=("Arial", 14))
        self.separator2.pack(anchor="w", pady=5)

        # ========================= YAWN AND BLINK SECTION ========================= #
        yb_frame = tk.Frame(text_frame)
        yb_frame.grid(row=3, column=0, sticky="w")
        self.yawnandblink_label = tk.Label(yb_frame, text="YAWN AND BLINK DETECTION", font=("Arial", 11, "bold"))
        self.yawnandblink_label.pack(anchor="w")

        self.blinks_label = tk.Label(yb_frame, text=f"Blinks: {yb.num_of_blinks_gui}", font=("Arial", 11))
        self.blinks_label.pack(anchor="w")

        self.microsleep_label = tk.Label(yb_frame, text=f"Microsleep Duration: {yb.microsleep_duration_gui:.2f}s", font=("Arial", 12))
        self.microsleep_label.pack(anchor="w")

        self.yawns_label = tk.Label(yb_frame, text=f"Yawns: {yb.num_of_yawns_gui}", font=("Arial", 11))
        self.yawns_label.pack(anchor="w")

        self.yawn_duration_label = tk.Label(yb_frame, text=f"Yawn Duration: {yb.yawn_duration_gui:.2f}s", font=("Arial", 11))
        self.yawn_duration_label.pack(anchor="w")

        self.blinks_per_minute_label = tk.Label(yb_frame, text=f"Blinks Per Minute: {yb.blinks_per_minute_gui}", font=("Arial", 11))
        self.blinks_per_minute_label.pack(anchor="w")

        self.yawns_per_minute_label = tk.Label(yb_frame, text=f"Yawns Per Minute: {yb.yawns_per_minute_gui}", font=("Arial", 11))
        self.yawns_per_minute_label.pack(anchor="w")

        self.alert_label = tk.Label(yb_frame, text="", font=("Arial", 11, "bold"), fg="red")
        self.alert_label.pack(anchor="w")
        
        # Initialize the video capture with local video file
        self.cap = cv2.VideoCapture("/path/to/your/local/video.mp4")  # Update this path
        
        if not self.cap.isOpened():
            print("⚠️ Error: Could not open the video file!")
            sys.exit(1)
        else:
            print("✅ Video file successfully opened!")

        self.frame = None
        self.running = True

        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()

        # Create instances of both detection classes
        self.gaze_detector = GazeAndHeadDetection()
        self.yawn_blink_detector = DrowsinessDetector()

        # Create a label for displaying status
        self.status_label = tk.Label(root, text="Initializing...", font=("Arial", 14))
        self.status_label.pack()

        # Start processing in separate threads
        self.gaze_thread = threading.Thread(target=self.process_gaze_head, daemon=True)
        self.yawn_thread = threading.Thread(target=self.process_yawn_blink, daemon=True)

        self.gaze_thread.start()
        self.yawn_thread.start()

        # Start GUI update loop
        self.update_info()
        self.update_camera()

    def update_camera(self):
        """ Continuously updates the camera feed in the Tkinter GUI. """
        if self.frame is not None:
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.root.after(20, self.update_camera)

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
        """Updates the GUI labels based on the distraction status."""
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

        # Define colors
        dark_orange = "#FF8C00"
        grey = "#808080"
        warning_text = ""
        warning_color = "black"

        if up_flag_gui == 0:
            self.gaze_center_label.config(text=f"Position: setting baseline..", fg=grey)
            self.gaze_status_label.config(text=f"Gaze Status: setting baseline..", fg=grey)
            self.pitch_label.config(text=f"Pitch: setting baseline..", fg=grey)
            self.yaw_label.config(text=f"Yaw: setting baseline..", fg=grey)
            self.roll_label.config(text=f"Roll: setting baseline..", fg=grey)
            self.head_status_label.config(text=f"Head Status: setting baseline..", fg=grey)

        elif up_flag_gui == 1:
            self.gaze_center_label.config(text=f"Position: {up_gaze_gui}", fg="black")

            if up_gaze_status_gui == "NORMAL":
                self.gaze_status_label.config(text=f"Gaze Status: {up_gaze_status_gui} ✅", fg="green")
            else:
                self.gaze_status_label.config(text=f"Gaze Status: {up_gaze_status_gui} 🚨", fg="red")

            self.pitch_label.config(text=f"Pitch: {up_pitch_gui:.2f} deg", fg="black")
            self.yaw_label.config(text=f"Yaw: {up_yaw_gui:.2f} deg", fg="black")
            self.roll_label.config(text=f"Roll: {up_roll_gui:.2f} deg", fg="black")

            if up_head_status_gui == "NORMAL":
                self.head_status_label.config(text=f"Head Status: {up_head_status_gui} ✅", fg="green")
            else:
                self.head_status_label.config(text=f"Head Status: {up_head_status_gui} 🚨", fg="red")

            self.distraction_label.config(text=f"Distraction Count within 3 min: {up_distraction_counter}", fg="black")

            if up_distraction_flag_gaze == 2 and up_temp_g == 1:
                warning_text += "🚨 HIGH RISK 🚨"
                warning_color = "red"
            elif up_distraction_flag_gaze == 1 and up_temp_g == 0:
                warning_text += "⚠️WARNING: Driver Distracted!"
                warning_color = dark_orange

            if up_distraction_flag_head == 2 and up_temp == 1:
                warning_text += "🚨 HIGH RISK 🚨"
                warning_color = "red"
            elif up_distraction_flag_head == 1 and up_temp == 0:
                warning_text += "⚠️WARNING: Driver Distracted!"
                warning_color = dark_orange

            if warning_text == "":
                self.distraction_flag_label.config(text="")
            else:
                self.distraction_flag_label.config(text=warning_text.strip(), fg=warning_color)

        # Update yawn and blink information
        self.blinks_label.config(text=f"num of blinks: {up_num_of_blinks_gui}", fg="black")
        self.microsleep_label.config(text=f"microsleep duration: {up_microsleep_duration_gui:.2f} sec", fg="black")
        self.yawns_label.config(text=f"num of yawns: {up_num_of_yawns_gui}", fg="black")
        self.yawn_duration_label.config(text=f"yawn duration: {up_yawn_duration_gui:.2f} sec", fg="black")
        self.blinks_per_minute_label.config(text=f"blinks/min: {up_blinks_per_minute_gui} ", fg="black")
        self.yawns_per_minute_label.config(text=f"yawns/min: {up_yawns_per_minute_gui} ", fg="black")

        # Handle fatigue warnings
        fatigue_alert_text = ""
        fatigue_alert_color = "black"
        
        if round(up_microsleep_duration_gui, 2) > microsleep_threshold:
            fatigue_alert_text = "⚠️Alert: Prolonged Microsleep Detected!"
            fatigue_alert_color = "red"
        elif up_gaze_gui == "Down" and up_gaze_status_gui == "ABNORMAL GAZE" and up_head_status_gui == "ABNORMAL PITCH":
            fatigue_alert_text = "⚠️Alert! Driver is fainted :("
            fatigue_alert_color = "red"
        elif round(up_yawn_duration_gui, 2) > yawning_threshold:
            fatigue_alert_text = "⚠️Alert: Prolonged Yawn Detected!"
            fatigue_alert_color = "orange"
        elif up_microsleep_duration_gui > microsleep_threshold:
            fatigue_alert_text = "⚠️Alert! Possible Fatigue!"
            fatigue_alert_color = "red"
        elif up_blinks_per_minute_gui > 35 or up_yawns_per_minute_gui > 5:
            fatigue_alert_text = "⚠️Alert! Driver is Highly Fatigued!"
            fatigue_alert_color = "red"
        elif up_blinks_per_minute_gui > 25 or up_yawns_per_minute_gui > 3:
            fatigue_alert_text = "⚠️ Alert! Driver is Possibly Fatigued!"
            fatigue_alert_color = "orange"

        self.alert_label.config(text=fatigue_alert_text, fg=fatigue_alert_color if fatigue_alert_text else "black")

        # Update status file
        self.update_status_file_camera2()
        
        # Schedule next update
        self.root.after(500, self.update_info)

    def capture_frames(self):
        """ Continuously captures frames from the video file """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                print("Frame Captured!")
            else:
                print("Failed to capture frame!")
                # If video ends, restart from beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def update_status_file_camera2(self):
        status_camera2 = {
            "gaze_center": self.gaze_center_label.cget("text"),
            "gaze_status": self.gaze_status_label.cget("text"),
            "pitch": self.pitch_label.cget("text"),
            "yaw": self.yaw_label.cget("text"),
            "roll": self.roll_label.cget("text"),
            "head_status": self.head_status_label.cget("text"),
            "distraction": self.distraction_label.cget("text"),
            "blinks": self.blinks_label.cget("text"),
            "microsleep_duration": self.microsleep_label.cget("text"),
            "yawns": self.yawns_label.cget("text"),
            "yawn_duration": self.yawn_duration_label.cget("text"),
            "blinks_per_minute": self.blinks_per_minute_label.cget("text"),
            "yawns_per_minute": self.yawns_per_minute_label.cget("text"),
            "alert": self.alert_label.cget("text")
        }
        
        try:
            with open("status_driver_fatigue.json", "w") as f:
                json.dump(status_camera2, f)
        except Exception as e:
            print("Error writing status.json:", e)

    def close_app(self):
        """ Release resources when closing the app """
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DriverMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop() 