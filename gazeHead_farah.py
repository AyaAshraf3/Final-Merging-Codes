import cv2  # OpenCV for video processing
import mediapipe as mp  # Mediapipe for face mesh and landmark detection
import numpy as np  # NumPy for mathematical calculations
import time  # Time module for timers
import platform
from threading import Thread

import tkinter as tk
from PIL import Image, ImageTk
from yawnBlink_farah import DrowsinessDetector
import yawnBlink_farah as yb  # Import the script to access its global variables
from thresholds import *
import importlib

importlib.reload(yb)  # ✅ This forces Python to reload the latest changes




# Global variables for GUI updates
pitch_gui = 0.0
yaw_gui = 0.0
roll_gui = 0.0
gaze_gui = "Center"
gaze_status_gui = 0
head_status_gui = 0
flag_gui = 0
distraction_flag_head = 0
distraction_flag_gaze = 0
temp = 0
temp_g = 0
distraction_counter = 0
gaze_flag=False
buzzer_running = False





class GazeAndHeadDetection:
    frame = None

    # Timers and state variables for baseline establishment
    start_time = time.time()  # Start time for collecting baseline head movement data
    threshold_time = 20  # Time in seconds to collect baseline data
    baseline_pitch, baseline_yaw, baseline_roll = 0, 0, 0  # Stores the average baseline angles
    baseline_data = []  # List to accumulate head movement angles during baseline collection
    baseline_set = False  # Flag to check if baseline has been established

    # Variables for detecting gaze and head movement abnormalities
    gaze_start_time = None  # Tracks the start time for abnormal gaze detection
    gaze_alert_triggered = False  # Flag indicating an abnormal gaze event
    gaze_abnormal_duration = 5  # Duration threshold (seconds) to trigger an abnormal gaze alert

    head_alert_start_time = None  # Tracks the start time for abnormal head movement detection
    head_alert_triggered = False  # Flag indicating an abnormal head movement event
    head_abnormal_duration = 5  # Duration threshold (seconds) to trigger an abnormal head movement alert

    # Thresholds for detecting abnormal head movements
    PITCH_THRESHOLD = 10  # Maximum allowed deviation in pitch angle (degrees)
    YAW_THRESHOLD = 10  # Maximum allowed deviation in yaw angle (degrees)
    ROLL_THRESHOLD = 10  # Maximum allowed deviation in roll angle (degrees)
    EAR_THRESHOLD = 0.35  # Eye Aspect Ratio (EAR) threshold below which gaze is considered downward
    NO_BLINK_GAZE_DURATION_INTIAL = 10  # Time (seconds) for which continuous center gaze without blinking is abnormal

    # Buzzer control for alerts
    
    no_blink_start_time = None  # Tracks time since last detected blink

    # Distraction tracking variables
    start_time_counter = time.time()  # Start time for tracking distractions
    DISTRACTION_THRESHOLD = 4  # Number of distractions before issuing a warning

    # Flag to ensure each abnormal gaze is counted only once
    #gaze_flag = False

    def __init__(self):
        global pitch_gui, yaw_gui, roll_gui, gaze_gui
        global gaze_status_gui, head_status_gui, flag_gui
        global distraction_flag_head, distraction_flag_gaze, temp, temp_g, distraction_counter
        global gaze_flag

        # Initialize Mediapipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  
            refine_landmarks=True,  # ✅ Enable iris landmarks (468-471)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_frame(self, frame):
        global pitch_gui, yaw_gui, roll_gui, gaze_gui
        global gaze_status_gui, head_status_gui, flag_gui
        global distraction_flag_head, distraction_flag_gaze, temp, temp_g, distraction_counter
        global gaze_flag

        #print("dakhal el process frame")
        GazeAndHeadDetection.frame = frame
        # Calculate elapsed time for distraction tracking
        elapsed_time_counter = time.time() - GazeAndHeadDetection.start_time_counter

        # Convert frame to RGB format for Mediapipe processing
        h, w, _ = GazeAndHeadDetection.frame.shape
        rgb_frame = cv2.cvtColor(GazeAndHeadDetection.frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)  # Process frame using Mediapipe Face Mesh

        if results.multi_face_landmarks:  # If a face is detected
            for face_landmarks in results.multi_face_landmarks:

                # ✅ Step 1: Collect baseline angles if not set
                if not GazeAndHeadDetection.baseline_set:
                    pitch, yaw, roll = self.calculate_angles(face_landmarks.landmark, w, h)
                    elapsed_time = time.time() - GazeAndHeadDetection.start_time  # Track baseline time

                    GazeAndHeadDetection.baseline_data.append((pitch, yaw, roll))  # Store collected angles
                    
                    if elapsed_time >= GazeAndHeadDetection.threshold_time:
                        # Compute baseline averages
                        GazeAndHeadDetection.baseline_pitch, GazeAndHeadDetection.baseline_yaw, GazeAndHeadDetection.baseline_roll = np.mean(
                            GazeAndHeadDetection.baseline_data, axis=0)
                        GazeAndHeadDetection.baseline_set = True  # Mark baseline as set
                        

                else:
                    # ✅ Step 2: Calculate head angles
                    

                    flag_gui = 1  # Enable GUI updates
                    #GazeAndHeadDetection.pitch, GazeAndHeadDetection.yaw, GazeAndHeadDetection.roll = GazeAndHeadDetection.calculate_angles(
                       # face_landmarks.landmark, w, h)
                    pitch_t, yaw_t, roll_t = self.calculate_angles(face_landmarks.landmark, w, h)

                    # ✅ Print the values
                    

                    # ✅ Assign them to class variables
                    GazeAndHeadDetection.pitch = pitch_t
                    GazeAndHeadDetection.yaw = yaw_t
                    GazeAndHeadDetection.roll = roll_t
                    pitch_gui, yaw_gui, roll_gui = GazeAndHeadDetection.pitch, GazeAndHeadDetection.yaw, GazeAndHeadDetection.roll  # Update GUI values
                    
                    # Detect abnormal movements
                    head_alerts = []
                    if abs(GazeAndHeadDetection.pitch - GazeAndHeadDetection.baseline_pitch) > GazeAndHeadDetection.PITCH_THRESHOLD or GazeAndHeadDetection.pitch > 73:
                        head_alerts = self.check_abnormal_angles(GazeAndHeadDetection.pitch, GazeAndHeadDetection.yaw, GazeAndHeadDetection.roll, 'pitch')
                    if abs(GazeAndHeadDetection.yaw - GazeAndHeadDetection.baseline_yaw) > GazeAndHeadDetection.YAW_THRESHOLD:
                        head_alerts = self.check_abnormal_angles(GazeAndHeadDetection.pitch, GazeAndHeadDetection.yaw, GazeAndHeadDetection.roll, 'yaw')
                    if abs(GazeAndHeadDetection.roll - GazeAndHeadDetection.baseline_roll) > GazeAndHeadDetection.ROLL_THRESHOLD:
                        head_alerts = self.check_abnormal_angles(GazeAndHeadDetection.pitch, GazeAndHeadDetection.yaw, GazeAndHeadDetection.roll, 'roll')

                    # ✅ Step 3: Trigger alerts based on detected movements
                    if head_alerts:
                        if GazeAndHeadDetection.head_alert_start_time is None:  # Start timer
                            GazeAndHeadDetection.head_alert_start_time = time.time()
                        elif time.time() - GazeAndHeadDetection.head_alert_start_time > GazeAndHeadDetection.head_abnormal_duration and not GazeAndHeadDetection.head_alert_triggered:
                            GazeAndHeadDetection.head_alert_triggered = True
                            distraction_counter += 1  # Increase distraction counter
                    else:
                        GazeAndHeadDetection.head_alert_start_time = None  # Reset timer
                        GazeAndHeadDetection.head_alert_triggered = False

                    # ✅ Step 4: Update GUI warnings
                    if GazeAndHeadDetection.head_alert_triggered:
                        if "Abnormal Pitch" in head_alerts:
                            head_status_gui = "ABNORMAL PITCH"
                        elif "Abnormal Yaw" in head_alerts:
                            head_status_gui = "ABNORMAL YAW"
                        elif "Abnormal Roll" in head_alerts:
                            head_status_gui = "ABNORMAL ROLL"
                        distraction_flag_head = 1
                    else:
                        head_status_gui = "NORMAL"
                        distraction_flag_head = 0

        # -------------------------------------------- Gaze Detection --------------------------------------------
            # Define facial landmarks for eye and iris detection
            left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
            left_iris_indices = [468, 469, 470, 471]

            # Helper function to calculate the center of landmarks
            def get_center(landmarks, indices):
                points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
                return np.mean(points, axis=0)  # Compute mean center

            # Compute eye and iris centers
            left_eye_center = get_center(face_landmarks.landmark, left_eye_indices)
            left_iris_center = get_center(face_landmarks.landmark, left_iris_indices)

            # Compute iris horizontal position
            left_eye_width = np.linalg.norm(
                np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]) - 
                np.array([face_landmarks.landmark[133].x, face_landmarks.landmark[133].y])
            )
            left_iris_position_x = (left_iris_center[0] - left_eye_center[0]) / left_eye_width

            # Compute iris vertical position
            left_eye_height = np.linalg.norm(
                np.array([face_landmarks.landmark[159].x, face_landmarks.landmark[159].y]) - 
                np.array([face_landmarks.landmark[145].x, face_landmarks.landmark[145].y])
            )
            left_iris_position_y = (left_iris_center[1] - left_eye_center[1]) / left_eye_height

            # ✅ Step 1: Detect Gaze Direction
            if left_iris_position_x < -0.1:
                gaze = "Right"
        
            elif left_iris_position_x > 0.1:
                gaze = "Left"
            
            else:
                gaze = self.process_blink_and_gaze("Center", self.compute_ear(face_landmarks.landmark, left_eye_indices), left_iris_position_y)

            gaze_gui = gaze  # Update GUI

            # ✅ Step 2: Detect Abnormal Gaze
            if gaze in ["Left", "Right", "Down", "Center Gazed"]:
                if GazeAndHeadDetection.gaze_start_time is None:
                    GazeAndHeadDetection.gaze_start_time = time.time()
                elif time.time() - GazeAndHeadDetection.gaze_start_time > GazeAndHeadDetection.gaze_abnormal_duration and not GazeAndHeadDetection.gaze_alert_triggered:
                    GazeAndHeadDetection.gaze_alert_triggered = True

                    if not gaze_flag:
                        distraction_counter += 1
                        gaze_flag = True
            else:
                GazeAndHeadDetection.gaze_start_time = None
                GazeAndHeadDetection.gaze_alert_triggered = False

                if gaze == "Center":
                    gaze_flag = False

            # ✅ Step 3: Update Gaze Warnings
            if GazeAndHeadDetection.gaze_alert_triggered:
                gaze_status_gui = "ABNORMAL GAZE"
                distraction_flag_gaze = 1
                   
            else:
                gaze_status_gui = "NORMAL"
                distraction_flag_gaze = 0
                

            # -------------------------------------------- Distraction Handling --------------------------------------------
            # ✅ If distraction threshold is reached, trigger HIGH RISK alert
            if distraction_counter >= 4 and elapsed_time_counter < 180:
                temp = 1
                temp_g = 1
                distraction_flag_head = 2
                distraction_flag_gaze = 2

                # Activate buzzer alert
                self.buzzer_alert()
                  

                # Function to stop the buzzer after 5 seconds
                Thread(target=lambda: (time.sleep(4), self.stop_buzzer())).start()

                # Function to reset distraction flag after 9 seconds
                Thread(target=lambda: (time.sleep(7), self.reset_distraction_flag())).start()


            elif elapsed_time_counter >= 180:  # Reset counter every 3 minutes
                print("⏳ 3 minutes passed. Resetting counter.")
                distraction_counter = 0
                GazeAndHeadDetection.start_time_counter = time.time()

        # ✅ Display the processed frame
        #cv2.imshow("Head Movement and Gaze Detection", frame)

        # Exit loop on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        
    def calculate_pitch(self, nose, chin):
        """Compute the pitch angle using nose and chin landmarks."""
        # Vector from nose to chin (3D)
        # Positive value means the chin is lower than the nose (head tilted downward "forward")
        # Negative value means the chin is above the nose (head tilted upward "backward")
        vector = np.array([chin[0] - nose[0], chin[1] - nose[1], chin[2] - nose[2]])

        # Compute pitch using atan2 (Preserves sign for up/down movement)
        # Projection of the vector in the X-Z plane "np.linalg.norm([vector[0], vector[2]])"
        pitch_angle = np.degrees(np.arctan2(vector[1], np.linalg.norm([vector[0], vector[2]])))

        # Adjust for backward movement
        if chin[2] > nose[2]:  # If chin is deeper into the screen than the nose
            pitch_angle *= -1  # Invert pitch to reflect backward movement correctly

        return pitch_angle

    def compute_ear(self, landmarks, eye_indices):
        """Compute Eye Aspect Ratio (EAR) for detecting blinks."""
        # Vertical distances
        # Lower and upper eyelid of the left eye
        vertical1 = np.linalg.norm(
            np.array([landmarks[159].x, landmarks[159].y]) - 
            np.array([landmarks[145].x, landmarks[145].y])
        )
        # Lower and upper eyelid of the right eye
        vertical2 = np.linalg.norm(
            np.array([landmarks[158].x, landmarks[158].y]) - 
            np.array([landmarks[144].x, landmarks[144].y])
        )
        # Horizontal distance
        # Outer and inner corners of the left eye
        horizontal = np.linalg.norm(
            np.array([landmarks[33].x, landmarks[33].y]) - 
            np.array([landmarks[133].x, landmarks[133].y])
        )
        # Compute EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def process_blink_and_gaze(self, gaze, left_ear, left_iris_position_y):
        """Detect abnormal prolonged center gaze without blinking."""
        global no_blink_start_time, gaze_alert_triggered 

        if gaze == "Center":
            if GazeAndHeadDetection.no_blink_start_time is None:
                GazeAndHeadDetection.no_blink_start_time = time.time()
            else:
                elapsed_time = time.time() - GazeAndHeadDetection.no_blink_start_time
                if elapsed_time >= GazeAndHeadDetection.NO_BLINK_GAZE_DURATION_INTIAL:
                    gaze = "Center Gazed"  # Set gaze to "Center Gazed" to trigger alert
                       

        else:
            GazeAndHeadDetection.no_blink_start_time = None  # Reset timer if gaze changes

        if left_iris_position_y < -0.3 and left_ear < GazeAndHeadDetection.EAR_THRESHOLD:
            GazeAndHeadDetection.no_blink_start_time = None  # Reset blink timer if blink is detected
            gaze = "Down"
            
        return gaze

    def calculate_angles(self, landmarks, frame_width, frame_height):
        """Calculate pitch, yaw, and roll angles from facial landmarks."""
        # Select key points for calculating angles
        #print(" dakhal calcultae angle ")
        nose_tip = landmarks[1]  # Nose tip landmark
        chin = landmarks[152]  # Chin landmark
        left_eye_outer = landmarks[33]  # Outer corner of the left eye
        right_eye_outer = landmarks[263]  # Outer corner of the right eye
        forehead = landmarks[10]  # Forehead landmark

        # Convert normalized landmarks to pixel coordinates
        def normalized_to_pixel(normalized, width, height):
            return int(normalized.x * width), int(normalized.y * height)

        # Convert normalized coordinates to pixel coordinates for selected landmarks
        nose_tip = normalized_to_pixel(nose_tip, frame_width, frame_height)
        chin = normalized_to_pixel(chin, frame_width, frame_height)
        left_eye_outer = normalized_to_pixel(left_eye_outer, frame_width, frame_height)
        right_eye_outer = normalized_to_pixel(right_eye_outer, frame_width, frame_height)
        forehead = normalized_to_pixel(forehead, frame_width, frame_height)

        # Extract 3D coordinates (normalized) correctly using the passed `landmarks` variable
        nose_for_pitch = landmarks[1]
        chin_for_pitch = landmarks[152]

        # Convert to pixel coordinates
        nose_3d = np.array([nose_for_pitch.x * frame_width, nose_for_pitch.y * frame_height, nose_for_pitch.z * frame_width])
        chin_3d = np.array([chin_for_pitch.x * frame_width, chin_for_pitch.y * frame_height, chin_for_pitch.z * frame_width])

        # Compute pitch using 3D coordinates
        pitch = self.calculate_pitch(nose_3d, chin_3d)

        # Calculate yaw (horizontal head angle) based on the eye positions
        delta_x_eye = right_eye_outer[0] - left_eye_outer[0]
        delta_y_eye = right_eye_outer[1] - left_eye_outer[1]
        yaw = np.arctan2(delta_y_eye, delta_x_eye) * (180 / np.pi)

        # Calculate roll (head tilt angle) based on the forehead and chin positions
        delta_x_forehead = forehead[0] - chin[0]
        delta_y_forehead = forehead[1] - chin[1]
        roll = np.arctan2(delta_y_forehead, delta_x_forehead) * (180 / np.pi)

        return pitch, yaw, roll  # Return calculated angles

    def check_abnormal_angles(self, pitch, yaw, roll, movement_type):
        """Check if head movement angles exceed thresholds and return alerts."""
        alerts = []
        if movement_type == 'pitch':
            alerts.append("Abnormal Pitch")
        if movement_type == 'yaw':
            alerts.append("Abnormal Yaw")
        elif movement_type == 'roll':
            alerts.append("Abnormal Roll")
        return alerts

    def action_after_buzzer(self):
        global temp , temp_g
        temp=0
        temp_g=0

    def reset_distraction_flag(self):
        global distraction_flag_head, distraction_flag_gaze, distraction_counter, start_time_counter

        # Reset distraction flags
        distraction_flag_head = 0  
        distraction_flag_gaze = 0  

        # Reset distraction counter after buzzer stops
        distraction_counter = 0  
        start_time_counter = time.time()  

        # Run action_after_buzzer() after 1 second
        # Run `action_after_buzzer` after 1 second without blocking the GUI
        Thread(target=lambda: (time.sleep(1), self.action_after_buzzer())).start()

    def buzzer_alert(self):
        global buzzer_running
        if buzzer_running:
            return  # Prevent multiple buzzer threads
        buzzer_running = True
    
        def play_buzzer():
            while buzzer_running:
                if platform.system() == "Windows":
                    import winsound
                    winsound.Beep(1000, 500)  # Frequency 1000 Hz, Duration 500 ms
                else:
                    import os
                    os.system('play -nq -t alsa synth 0.5 sine 500')  # Works on Linux with sox installed
        
        buzzer_thread = Thread(target=play_buzzer, daemon=True)
        buzzer_thread.start()

    def stop_buzzer(self):
        global buzzer_running
        buzzer_running = False



