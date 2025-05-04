import cv2
import mediapipe as mp
import numpy as np
import time
import platform
from threading import Thread
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import tkinter as tk
from PIL import Image, ImageTk
from yawnBlink_farah_trt import DrowsinessDetector
import yawnBlink_farah_trt as yb
from thresholds import *
import importlib

importlib.reload(yb)

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
gaze_flag = False
buzzer_running = False

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input and output dimensions
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)
        
        # Allocate memory
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def preprocess(self, frame):
        # Resize and normalize
        frame = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        return frame

    def infer(self, frame):
        # Preprocess
        preprocessed = self.preprocess(frame)
        
        # Copy input to GPU
        np.copyto(self.inputs[0]['host'], preprocessed.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Copy output from GPU
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        
        # Process output
        output = self.outputs[0]['host'].reshape(self.output_shape)
        return output

class GazeAndHeadDetection:
    frame = None

    # Timers and state variables for baseline establishment
    start_time = time.time()
    threshold_time = 20
    baseline_pitch, baseline_yaw, baseline_roll = 0, 0, 0
    baseline_data = []
    baseline_set = False

    # Variables for detecting gaze and head movement abnormalities
    gaze_start_time = None
    gaze_alert_triggered = False
    gaze_abnormal_duration = 5

    head_alert_start_time = None
    head_alert_triggered = False
    head_abnormal_duration = 5

    # Thresholds for detecting abnormal head movements
    PITCH_THRESHOLD = 10
    YAW_THRESHOLD = 10
    ROLL_THRESHOLD = 10
    EAR_THRESHOLD = 0.35
    NO_BLINK_GAZE_DURATION_INTIAL = 10

    no_blink_start_time = None

    # Distraction tracking variables
    start_time_counter = time.time()
    DISTRACTION_THRESHOLD = 4

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
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize TensorRT model for face detection
        self.face_detector = TRTInference("face_detection_model.engine")
        
    def process_frame(self, frame):
        global pitch_gui, yaw_gui, roll_gui, gaze_gui
        global gaze_status_gui, head_status_gui, flag_gui
        global distraction_flag_head, distraction_flag_gaze, temp, temp_g, distraction_counter
        global gaze_flag

        GazeAndHeadDetection.frame = frame
        elapsed_time_counter = time.time() - GazeAndHeadDetection.start_time_counter

        # Convert frame to RGB format for Mediapipe processing
        h, w, _ = GazeAndHeadDetection.frame.shape
        rgb_frame = cv2.cvtColor(GazeAndHeadDetection.frame, cv2.COLOR_BGR2RGB)
        
        # Run face detection using TensorRT
        face_detection = self.face_detector.infer(rgb_frame)
        
        if face_detection is not None:  # If a face is detected
            # Process face landmarks using Mediapipe
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Rest of the processing remains the same as original
                    if not GazeAndHeadDetection.baseline_set:
                        pitch, yaw, roll = self.calculate_angles(face_landmarks.landmark, w, h)
                        elapsed_time = time.time() - GazeAndHeadDetection.start_time

                        GazeAndHeadDetection.baseline_data.append((pitch, yaw, roll))
                        
                        if elapsed_time >= GazeAndHeadDetection.threshold_time:
                            GazeAndHeadDetection.baseline_pitch, GazeAndHeadDetection.baseline_yaw, GazeAndHeadDetection.baseline_roll = np.mean(
                                GazeAndHeadDetection.baseline_data, axis=0)
                            GazeAndHeadDetection.baseline_set = True

                    else:
                        flag_gui = 1
                        pitch_t, yaw_t, roll_t = self.calculate_angles(face_landmarks.landmark, w, h)

                        GazeAndHeadDetection.pitch = pitch_t
                        GazeAndHeadDetection.yaw = yaw_t
                        GazeAndHeadDetection.roll = roll_t
                        pitch_gui, yaw_gui, roll_gui = GazeAndHeadDetection.pitch, GazeAndHeadDetection.yaw, GazeAndHeadDetection.roll

                        # Rest of the processing remains the same as original
                        head_alerts = []
                        if abs(GazeAndHeadDetection.pitch - GazeAndHeadDetection.baseline_pitch) > GazeAndHeadDetection.PITCH_THRESHOLD or GazeAndHeadDetection.pitch > 73:
                            head_alerts = self.check_abnormal_angles(GazeAndHeadDetection.pitch, GazeAndHeadDetection.yaw, GazeAndHeadDetection.roll, 'pitch')
                        if abs(GazeAndHeadDetection.yaw - GazeAndHeadDetection.baseline_yaw) > GazeAndHeadDetection.YAW_THRESHOLD:
                            head_alerts = self.check_abnormal_angles(GazeAndHeadDetection.pitch, GazeAndHeadDetection.yaw, GazeAndHeadDetection.roll, 'yaw')
                        if abs(GazeAndHeadDetection.roll - GazeAndHeadDetection.baseline_roll) > GazeAndHeadDetection.ROLL_THRESHOLD:
                            head_alerts = self.check_abnormal_angles(GazeAndHeadDetection.pitch, GazeAndHeadDetection.yaw, GazeAndHeadDetection.roll, 'roll')

                        # Rest of the code remains the same as original
                        if head_alerts:
                            if GazeAndHeadDetection.head_alert_start_time is None:
                                GazeAndHeadDetection.head_alert_start_time = time.time()
                            elif time.time() - GazeAndHeadDetection.head_alert_start_time > GazeAndHeadDetection.head_abnormal_duration and not GazeAndHeadDetection.head_alert_triggered:
                                GazeAndHeadDetection.head_alert_triggered = True
                                distraction_counter += 1
                        else:
                            GazeAndHeadDetection.head_alert_start_time = None
                            GazeAndHeadDetection.head_alert_triggered = False

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

                        # Gaze detection code remains the same
                        left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
                        left_iris_indices = [468, 469, 470, 471]

                        def get_center(landmarks, indices):
                            points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
                            return np.mean(points, axis=0)

                        left_eye_center = get_center(face_landmarks.landmark, left_eye_indices)
                        left_iris_center = get_center(face_landmarks.landmark, left_iris_indices)

                        left_eye_width = np.linalg.norm(
                            np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]) - 
                            np.array([face_landmarks.landmark[133].x, face_landmarks.landmark[133].y])
                        )
                        left_iris_position_x = (left_iris_center[0] - left_eye_center[0]) / left_eye_width

                        left_eye_height = np.linalg.norm(
                            np.array([face_landmarks.landmark[159].x, face_landmarks.landmark[159].y]) - 
                            np.array([face_landmarks.landmark[145].x, face_landmarks.landmark[145].y])
                        )
                        left_iris_position_y = (left_iris_center[1] - left_eye_center[1]) / left_eye_height

                        if left_iris_position_x < -0.1:
                            gaze = "Right"
                        elif left_iris_position_x > 0.1:
                            gaze = "Left"
                        else:
                            gaze = self.process_blink_and_gaze("Center", self.compute_ear(face_landmarks.landmark, left_eye_indices), left_iris_position_y)

                        gaze_gui = gaze

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

                        if GazeAndHeadDetection.gaze_alert_triggered:
                            gaze_status_gui = "ABNORMAL GAZE"
                            distraction_flag_gaze = 1
                        else:
                            gaze_status_gui = "NORMAL"
                            distraction_flag_gaze = 0

                        if distraction_counter >= 4 and elapsed_time_counter < 180:
                            temp = 1
                            temp_g = 1
                            distraction_flag_head = 2
                            distraction_flag_gaze = 2

                            self.buzzer_alert()
                            Thread(target=lambda: (time.sleep(4), self.stop_buzzer())).start()
                            Thread(target=lambda: (time.sleep(7), self.reset_distraction_flag())).start()

                        elif elapsed_time_counter >= 180:
                            print("⏳ 3 minutes passed. Resetting counter.")
                            distraction_counter = 0
                            GazeAndHeadDetection.start_time_counter = time.time()

    # Rest of the methods remain the same as original
    def calculate_pitch(self, nose, chin):
        vector = np.array([chin[0] - nose[0], chin[1] - nose[1], chin[2] - nose[2]])
        pitch_angle = np.degrees(np.arctan2(vector[1], np.linalg.norm([vector[0], vector[2]])))
        if chin[2] > nose[2]:
            pitch_angle *= -1
        return pitch_angle

    def compute_ear(self, landmarks, eye_indices):
        vertical1 = np.linalg.norm(
            np.array([landmarks[159].x, landmarks[159].y]) - 
            np.array([landmarks[145].x, landmarks[145].y])
        )
        vertical2 = np.linalg.norm(
            np.array([landmarks[158].x, landmarks[158].y]) - 
            np.array([landmarks[144].x, landmarks[144].y])
        )
        horizontal = np.linalg.norm(
            np.array([landmarks[33].x, landmarks[33].y]) - 
            np.array([landmarks[133].x, landmarks[133].y])
        )
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def process_blink_and_gaze(self, gaze, left_ear, left_iris_position_y):
        if gaze == "Center":
            if GazeAndHeadDetection.no_blink_start_time is None:
                GazeAndHeadDetection.no_blink_start_time = time.time()
            else:
                elapsed_time = time.time() - GazeAndHeadDetection.no_blink_start_time
                if elapsed_time >= GazeAndHeadDetection.NO_BLINK_GAZE_DURATION_INTIAL:
                    gaze = "Center Gazed"
        else:
            GazeAndHeadDetection.no_blink_start_time = None

        if left_iris_position_y < -0.3 and left_ear < GazeAndHeadDetection.EAR_THRESHOLD:
            GazeAndHeadDetection.no_blink_start_time = None
            gaze = "Down"
            
        return gaze

    def calculate_angles(self, landmarks, frame_width, frame_height):
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        forehead = landmarks[10]

        def normalized_to_pixel(normalized, width, height):
            return int(normalized.x * width), int(normalized.y * height)

        nose_tip = normalized_to_pixel(nose_tip, frame_width, frame_height)
        chin = normalized_to_pixel(chin, frame_width, frame_height)
        left_eye_outer = normalized_to_pixel(left_eye_outer, frame_width, frame_height)
        right_eye_outer = normalized_to_pixel(right_eye_outer, frame_width, frame_height)
        forehead = normalized_to_pixel(forehead, frame_width, frame_height)

        nose_for_pitch = landmarks[1]
        chin_for_pitch = landmarks[152]

        nose_3d = np.array([nose_for_pitch.x * frame_width, nose_for_pitch.y * frame_height, nose_for_pitch.z * frame_width])
        chin_3d = np.array([chin_for_pitch.x * frame_width, chin_for_pitch.y * frame_height, chin_for_pitch.z * frame_width])

        pitch = self.calculate_pitch(nose_3d, chin_3d)

        delta_x_eye = right_eye_outer[0] - left_eye_outer[0]
        delta_y_eye = right_eye_outer[1] - left_eye_outer[1]
        yaw = np.arctan2(delta_y_eye, delta_x_eye) * (180 / np.pi)

        delta_x_forehead = forehead[0] - chin[0]
        delta_y_forehead = forehead[1] - chin[1]
        roll = np.arctan2(delta_y_forehead, delta_x_forehead) * (180 / np.pi)

        return pitch, yaw, roll

    def check_abnormal_angles(self, pitch, yaw, roll, movement_type):
        alerts = []
        if movement_type == 'pitch':
            alerts.append("Abnormal Pitch")
        if movement_type == 'yaw':
            alerts.append("Abnormal Yaw")
        elif movement_type == 'roll':
            alerts.append("Abnormal Roll")
        return alerts

    def action_after_buzzer(self):
        global temp, temp_g
        temp = 0
        temp_g = 0

    def reset_distraction_flag(self):
        global distraction_flag_head, distraction_flag_gaze, distraction_counter, start_time_counter
        distraction_flag_head = 0
        distraction_flag_gaze = 0
        distraction_counter = 0
        start_time_counter = time.time()
        Thread(target=lambda: (time.sleep(1), self.action_after_buzzer())).start()

    def buzzer_alert(self):
        global buzzer_running
        if buzzer_running:
            return
        buzzer_running = True
    
        def play_buzzer():
            while buzzer_running:
                if platform.system() == "Windows":
                    import winsound
                    winsound.Beep(1000, 500)
                else:
                    import os
                    os.system('play -nq -t alsa synth 0.5 sine 500')
        
        buzzer_thread = Thread(target=play_buzzer, daemon=True)
        buzzer_thread.start()

    def stop_buzzer(self):
        global buzzer_running
        buzzer_running = False 