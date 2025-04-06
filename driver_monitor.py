import threading
import cv2
import time
import sys
import tkinter as tk
from thresholds import *
from PIL import Image, ImageTk  
import importlib

from yawnBlink_farah import DrowsinessDetector
from gazeHead_farah import GazeAndHeadDetection  # Import your class
import os
import json
from flask import Flask, Response
#from shared_cameras import camera2 # Import the shared camera instance


import yawnBlink_farah as yb  # Import the script to access its global variables
importlib.reload(yb)  # ✅ This forces Python to reload the latest changes

import gazeHead_farah as gh  # Import the script to access its global variables
importlib.reload(gh)  # ✅ This forces Python to reload the latest changes



class DriverMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Monitoring System")
        
        # ========================= TITLE SECTION ========================= #
        self.title_label = tk.Label(
            root, text="Driver's Fatigue and Distraction Module",
            font=("Courier New", 22, "bold"), fg="blue"
        )
        self.title_label.pack(pady=10)  # Add padding below the title

        # ========================= MAIN CONTAINER ========================= #
        frame_container = tk.Frame(root)
        frame_container.pack(fill="both", expand=True, padx=10, pady=10)

        # ========================= VIDEO FEED SECTION (LEFT SIDE) ========================= #
        self.video_label = tk.Label(frame_container)
        self.video_label.grid(row=0, column=0, rowspan=10, padx=10, pady=10)  # Video feed label

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
        self.separator2.pack(anchor="w", pady=5)  # Separator


        # ========================= YAWN AND BLINK SECTION ========================= #
        yb_frame = tk.Frame(text_frame)
        yb_frame.grid(row=3, column=0, sticky="w")
        self.yawnandblink_label = tk.Label(yb_frame, text="YAWN AND BLINK DETECTION", font=("Arial", 11, "bold"))
        self.yawnandblink_label.pack(anchor="w")
        # Create labels for the drowsiness detection variables
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
        
        # Initialize the video capture
        self.cap = cv2.VideoCapture("http://127.0.0.1:5000/video_feed2")
        
        # Using the shared camera2 object
        #self.cap = camera2  

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
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (Tkinter)
            img = Image.fromarray(frame_rgb)  # Convert NumPy array to Image
            img = img.resize((640, 480))  # Resize to fit the GUI
            imgtk = ImageTk.PhotoImage(image=img)

            # Update video_label with the new frame
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        # Schedule the next frame update
        self.root.after(20, self.update_camera)  # Update every 20ms

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

        up_pitch_gui= gh.pitch_gui 
        up_yaw_gui = gh.yaw_gui 
        up_roll_gui= gh.roll_gui 
        up_gaze_gui= gh.gaze_gui 
        up_gaze_status_gui = gh.gaze_status_gui 
        up_head_status_gui= gh.head_status_gui 
        up_flag_gui = gh.flag_gui
        up_distraction_flag_head= gh.distraction_flag_head 
        up_distraction_flag_gaze= gh.distraction_flag_gaze 
        up_temp= gh.temp 
        up_temp_g= gh.temp_g 
        up_distraction_counter= gh.distraction_counter 
        up_gaze_flag= gh.gaze_flag
        up_buzzer_running=gh.buzzer_running 

        up_num_of_blinks_gui= yb.num_of_blinks_gui 
        up_microsleep_duration_gui=yb.microsleep_duration_gui 
        up_num_of_yawns_gui=yb.num_of_yawns_gui 
        up_yawn_duration_gui=yb.yawn_duration_gui 
        up_blinks_per_minute_gui=yb.blinks_per_minute_gui 
        up_yawns_per_minute_gui=yb.yawns_per_minute_gui

        # Define colors
        dark_orange = "#FF8C00"  # Warning color
        grey = "#808080"  # Default color for baseline setting
        warning_text = ""  # Initialize empty warning message
        warning_color = "black"  # Default text color

        ### 📌 Baseline Setup (Before Detection Starts) ###
        if up_flag_gui == 0:
            # Display "Setting Baseline..." for all metrics


            # Update GUI labels
            self.gaze_center_label.config(text=f"Position: setting baseline..", fg=grey)
            self.gaze_status_label.config(text=f"Gaze Status: setting baseline..", fg=grey)
            self.pitch_label.config(text=f"Pitch: setting baseline..", fg=grey)
            self.yaw_label.config(text=f"Yaw: setting baseline..", fg=grey)
            self.roll_label.config(text=f"Roll: setting baseline..", fg=grey)
            self.head_status_label.config(text=f"Head Status: setting baseline..", fg=grey)

        ### 📌 Detection & Warnings (Active Monitoring) ###
        elif up_flag_gui == 1:
            # Update gaze position
            self.gaze_center_label.config(text=f"Position: {up_gaze_gui}", fg="black")

            # Display gaze status (Normal / Abnormal)
            if up_gaze_status_gui == "NORMAL":
                self.gaze_status_label.config(text=f"Gaze Status: {up_gaze_status_gui} ✅", fg="green")  # Normal gaze
            else:
                self.gaze_status_label.config(text=f"Gaze Status: {up_gaze_status_gui} 🚨", fg="red")  # Abnormal gaze

            # Update head movement values
            self.pitch_label.config(text=f"Pitch: {up_pitch_gui:.2f} deg", fg="black")
            self.yaw_label.config(text=f"Yaw: {up_yaw_gui:.2f} deg", fg="black")
            self.roll_label.config(text=f"Roll: {up_roll_gui:.2f} deg", fg="black")

            # Display head status (Normal / Abnormal)
            if up_head_status_gui == "NORMAL":
                self.head_status_label.config(text=f"Head Status: {up_head_status_gui} ✅", fg="green")
            else:
                self.head_status_label.config(text=f"Head Status: {up_head_status_gui} 🚨", fg="red")

            # Update distraction counter
            self.distraction_label.config(text=f"Distraction Count within 3 min: {up_distraction_counter}", fg="black")

            ### 📌 Checking Distraction Warnings ###

            # 🛑 Gaze Warnings
            if up_distraction_flag_gaze == 2 and up_temp_g == 1:  # High risk due to gaze
                warning_text += "🚨 HIGH RISK 🚨"
                warning_color = "red"
            elif up_distraction_flag_gaze == 1 and up_temp_g == 0:  # Moderate distraction due to gaze
                warning_text += "⚠️WARNING: Driver Distracted!"
                warning_color = dark_orange

            # 🛑 Head Movement Warnings
            if up_distraction_flag_head == 2 and up_temp == 1:  # High risk due to head movement
                warning_text += "🚨 HIGH RISK 🚨"
                warning_color = "red"
            elif up_distraction_flag_head == 1 and up_temp == 0:  # Moderate distraction due to head movement
                warning_text += "⚠️WARNING: Driver Distracted!"
                warning_color = dark_orange

            ### 📌 Updating the Warning Label ###
            if warning_text == "":
                self.distraction_flag_label.config(text="")  # No warning
            else:
                self.distraction_flag_label.config(text=warning_text.strip(), fg=warning_color)  # Display warning message


        #-----display yawn and blink-------#
        self.blinks_label.config(text=f"num of blinks: {up_num_of_blinks_gui}", fg="black")
        self.microsleep_label.config(text=f"microsleep duration: {up_microsleep_duration_gui:.2f} sec", fg="black")
        self.yawns_label.config(text=f"num of yawns: {up_num_of_yawns_gui}", fg="black")
        self.yawn_duration_label.config(text=f"yawn duration: {up_yawn_duration_gui:.2f} sec", fg="black")
        self.blinks_per_minute_label.config(text=f"blinks/min: {up_blinks_per_minute_gui} ", fg="black")
        self.yawns_per_minute_label.config(text=f"yawns/min: {up_yawns_per_minute_gui} ", fg="black")


        # ----- Handle Fatigue Warnings (Display in Same Alert Area) ----- #
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

        # Update alert area with fatigue warnings
        self.alert_label.config(text=fatigue_alert_text, fg=fatigue_alert_color if fatigue_alert_text else "black")

        # Call the JSON update for camera2:
        self.update_status_file_camera2()
        
        # 🔄 Schedule the function to run again in 500ms
        #self.root.after(500, lambda: threading.Thread(target=self.update_info, daemon=True).start())
        self.root.after(500, self.update_info)  # ✅ Run in main thread, no new threads
        



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
            "gaze_center": self.gaze_center_label.cget("text"),         # e.g., "Position: Center: ..."
            "gaze_status": self.gaze_status_label.cget("text"),             # e.g., "Status: NORMAL ✅" or "Status: ABNORMAL 🚨"
            "pitch": self.pitch_label.cget("text"),                         # e.g., "Pitch: 0.00 deg"
            "yaw": self.yaw_label.cget("text"),                             # e.g., "Yaw: 0.00 deg"
            "roll": self.roll_label.cget("text"),                           # e.g., "Roll: 0.00 deg"
            "head_status": self.head_status_label.cget("text"),             # e.g., "Head Status: NORMAL"
            "distraction": self.distraction_label.cget("text"),             # e.g., "Distraction Count within 3 min: 0"
            "blinks": self.blinks_label.cget("text"),                       # e.g., "num of blinks: 0"
            "microsleep_duration": self.microsleep_label.cget("text"),      # e.g., "microsleep duration: 0.00 sec"
            "yawns": self.yawns_label.cget("text"),                         # e.g., "num of yawns: 0"
            "yawn_duration": self.yawn_duration_label.cget("text"),         # e.g., "yawn duration: 0.00 sec"
            "blinks_per_minute": self.blinks_per_minute_label.cget("text"),   # e.g., "blinks/min: 0"
            "yawns_per_minute": self.yawns_per_minute_label.cget("text"),     # e.g., "yawns/min: 0"
            "alert": self.alert_label.cget("text")                          # e.g., any fatigue alert text
        }
        
        try:
            with open("status_driver_fatigue.json", "w") as f:
                json.dump(status_camera2, f)
        except Exception as e:
            print("Error writing status.json:", e)
            
        # filename = "status_driver_fatigue.json"
        # # Read the existing file if it exists so that we can preserve other sections (like camera1)
        # data = {}
        # if os.path.exists(filename):
        #     try:
        #         with open(filename, "r") as f:
        #             data = json.load(f)
        #     except Exception as e:
        #         print("Error reading existing status.json:", e)
        #         data = {}
        # # Update (or add) the camera2 section
        # data["camera2"] = status_camera2
        # try:
        #     with open(filename, "w") as f:
        #         json.dump(data, f)
        # except Exception as e:
        #     print("Error writing to status.json:", e)


    def close_app(self):
        """ Release resources when closing the app """
        self.cap.release()
        self.root.destroy()

# def generate_frames():
#     while True:
#         #ret, frame = self.cap.read()
#         # if not ret:
#         #     break
#         frame = camera2.get_frame()  # Use the shared camera2 instance
#         if frame is None:
#             continue  # Wait until a frame is available
        
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue
        
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         time.sleep(0.03)  # Adjust frame rate as needed
        
# app = Flask(__name__) # Flask app for serving the video feed
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def run_stream():
#     app.run(host='0.0.0.0', port=5001)

# Run the Tkinter GUI

if __name__ == "__main__":
    root = tk.Tk()
    app = DriverMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    # Start the stream server in a separate thread
    # stream_thread = threading.Thread(target=run_stream, daemon=True)
    # stream_thread.start()
    root.mainloop()
 