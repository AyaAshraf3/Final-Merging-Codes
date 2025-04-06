import queue  # Used for thread-safe frame buffering
import threading  # Handles video capture and processing in parallel
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import sys
from thresholds import *  # Import thresholds for blink and yawn detection
import time

# Global Variables for GUI
num_of_blinks_gui = 0
microsleep_duration_gui = 0
num_of_yawns_gui = 0
yawn_duration_gui = 0
blinks_per_minute_gui = 0
yawns_per_minute_gui = 0



class DrowsinessDetector(): 
    def __init__(self):
        super().__init__()

        # Store current states
        self.yawn_state = ''
        self.eyes_state = ''
        self.alert_text = ''

        # Track statistics
        self.num_of_blinks = 0
        self.microsleep_duration = 0
        self.num_of_yawns = 0
        self.yawn_duration = 0

        # Track blinks/yawns per minute
        self.blinks_per_minute = 0
        self.yawns_per_minute = 0
        self.current_blinks = 0
        self.current_yawns = 0
        self.time_window = 60  # 1-minute window
        self.start_time = time.time()  # Track start time

        self.eyes_still_closed = False  # Track closed-eye state
      
        # Initialize yawn-related tracking variables
        self.yawn_finished = False  # ✅ Add this to prevent AttributeError
        self.yawn_in_progress = False  # ✅ Track if yawning is ongoing

        # ✅ Store the latest frame globally within the class
        self.current_frame = None  

        # Load YOLO model
        self.detect_drowsiness = YOLO(r"D:\grad project\driver_fatigue\trained_weights\best_ours2.pt")
        self.detect_drowsiness.to('cuda')  # Use GPU for inference
        
        # Using Multi-Threading (Only for tracking blink/yawn rates)
        self.stop_event = threading.Event()
        self.blink_yawn_thread = threading.Thread(target=self.update_blink_yawn_rate)
        self.blink_yawn_thread.start()  # Start the blink/yawn tracking thread

    def predict(self):
        """Processes the current frame and returns the detected state for eyes and yawning."""
        if self.current_frame is None:
            return "No Detection"

        results = self.detect_drowsiness(self.current_frame)  # ✅ Ensure correct inference call
        if not results:  # ✅ Check if results exist
            return "No Detection"

        boxes = results[0].boxes if hasattr(results[0], 'boxes') else None
        if boxes is None or len(boxes) == 0:  # ✅ Ensure valid detection boxes
            return "No Detection"

        confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
        class_ids = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
        
        if len(confidences) == 0 or len(class_ids) == 0:  # ✅ Handle empty detection cases
            return "No Detection"

        max_confidence_index = np.argmax(confidences)  # ✅ Choose highest confidence detection
        class_id = int(class_ids[max_confidence_index])  # ✅ Get predicted class

        # ✅ Return classification based on class_id
        if class_id == 0:
            return "Opened Eye"
        elif class_id == 1:
            return "Closed Eye"
        elif class_id == 2:
            return "Yawning"
        else:
            return "No Yawn"


    def process_frames(self, frame):
        """Receives and stores the latest frame, then processes it for detection."""
        global num_of_blinks_gui, microsleep_duration_gui, num_of_yawns_gui
        global yawn_duration_gui, blinks_per_minute_gui, yawns_per_minute_gui

        # ✅ Store the latest frame globally inside the class
        self.current_frame = frame  

        try:
            self.eyes_state = self.predict()  # Predict using stored frame

            # Handle eye blink detection
            if self.eyes_state == "Closed Eye":
                if not self.eyes_still_closed:
                    self.eyes_still_closed = True
                    self.start = time.perf_counter()
                    self.num_of_blinks += 1
                    num_of_blinks_gui = self.num_of_blinks  # ✅ Update global variable
                    self.current_blinks += 1  # ✅ Accumulate correctly
                self.microsleep_duration = time.perf_counter() - self.start
                microsleep_duration_gui = self.microsleep_duration
            else:
                self.eyes_still_closed = False
                self.microsleep_duration = 0
                microsleep_duration_gui = self.microsleep_duration

            # ✅ Handle Yawn Detection (Fixed Logic)
            if self.eyes_state == "Yawning":
                if not self.yawn_in_progress:  # ✅ Detect new yawn
                    self.start = time.perf_counter()  # ✅ Start tracking yawn
                    self.yawn_in_progress = True  # ✅ Start tracking yawn
                    self.yawn_duration = 0  # ✅ Reset duration
                self.yawn_duration = time.perf_counter() - self.start  # ✅ Accumulate yawn duration
                yawn_duration_gui = self.yawn_duration  # ✅ Update GUI

                if yawn_duration_gui > yawning_threshold and not self.yawn_finished:
                    self.yawn_finished = True  # ✅ Mark yawn as finished
                    self.num_of_yawns += 1  # ✅ Increment the yawn count
                    num_of_yawns_gui = self.num_of_yawns  # ✅ Update GUI
                    self.current_yawns += 1  # ✅ Accumulate correctly
                    print(f"Yawn detected! Total Yawns: {self.num_of_yawns}")

            else:  # ✅ Reset only when yawning has ended
                if self.yawn_in_progress:
                    self.yawn_in_progress = False  # ✅ Reset state
                    self.yawn_finished = False  # ✅ Allow new yawns

                self.yawn_duration = 0  # ✅ Reset duration
                yawn_duration_gui = self.yawn_duration  # ✅ Update GUI

        except Exception as e:
            print(f"Error in processing the frame: {e}") 


    def update_blink_yawn_rate(self):
        """✅ Updates blink and yawn rates every minute, correctly tracking values."""
        global blinks_per_minute_gui, yawns_per_minute_gui

        while not self.stop_event.is_set():
            time.sleep(self.time_window)  # ✅ Wait for 1 minute
            self.blinks_per_minute = self.current_blinks
            blinks_per_minute_gui = self.blinks_per_minute  # ✅ Update GUI variable
            self.yawns_per_minute = self.current_yawns
            yawns_per_minute_gui = self.yawns_per_minute  # ✅ Update GUI variable

            print(f"Updated Rates - Blinks: {self.blinks_per_minute} per min, Yawns: {self.yawns_per_minute} per min")

            # ✅ Do not reset current values until next cycle
            self.current_blinks = 0
            self.current_yawns = 0

    def fatigue_detection(self):
        """Triggers alerts based on fatigue detection using the latest frame."""
        global possibly_fatigued_alert, highly_fatigued_alert, possible_fatigue_alert

        if self.current_frame is None:
            return  # No frame available, skip processing

        microsleep_duration = microsleep_duration_gui
        blink_rate = blinks_per_minute_gui
        yawning_rate = yawns_per_minute_gui

        #if microsleep_duration > microsleep_threshold:
         #   possible_fatigue_alert = 1
        #if blink_rate > 35 or yawning_rate > 5:
         #   highly_fatigued_alert = 1
        #elif blink_rate > 25 or yawning_rate > 3:
         #   possibly_fatigued_alert = 1

    def play_alert_sound(self):
        """Plays an alert sound for fatigue detection."""
        frequency = 1000
        duration = 500
        winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        """Runs the alert sound in a separate thread."""
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()
