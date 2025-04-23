import queue  # Used for thread-safe frame buffering
import threading  # Handles video capture and processing in parallel
import time
import winsound
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
from thresholds import *  # Import thresholds for blink and yawn detection

# Global Variables for GUI
num_of_blinks_gui = 0
microsleep_duration_gui = 0
num_of_yawns_gui = 0
yawn_duration_gui = 0
blinks_per_minute_gui = 0
yawns_per_minute_gui = 0

# TensorRT setup
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = r"D:\grad project\driver_fatigue\trained_weights\best_ours2.engine"  # Update this path

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

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
        self.yawn_finished = False
        self.yawn_in_progress = False

        # Store the latest frame globally within the class
        self.current_frame = None  

        # Load TensorRT engine
        self.engine = load_engine(ENGINE_PATH)
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

        # Get input dimensions
        self.binding_shape = self.engine.get_binding_shape(0)
        self.input_shape = (self.binding_shape[2], self.binding_shape[3])  # (height, width)

        # Using Multi-Threading (Only for tracking blink/yawn rates)
        self.stop_event = threading.Event()
        self.blink_yawn_thread = threading.Thread(target=self.update_blink_yawn_rate)
        self.blink_yawn_thread.start()  # Start the blink/yawn tracking thread

    def preprocess(self, frame):
        """Preprocess the frame for TensorRT inference."""
        # Resize and normalize
        img = cv2.resize(frame, self.input_shape)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        return img

    def predict(self):
        """Processes the current frame and returns the detected state for eyes and yawning."""
        if self.current_frame is None:
            return "No Detection"

        try:
            # Preprocess the frame
            img = self.preprocess(self.current_frame)

            # Copy input to device
            np.copyto(self.inputs[0][0], img.ravel())
            cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)

            # Execute inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # Copy output from device
            cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)
            self.stream.synchronize()

            # Process output
            output = self.outputs[0][0]
            output = output.reshape(1, -1, 6)  # Reshape to (batch, num_detections, 6)
            
            # Filter detections by confidence threshold
            detections = output[output[..., 4] > 0.5]  # Confidence threshold of 0.5
            
            if len(detections) == 0:
                return "No Detection"

            # Get the detection with highest confidence
            best_idx = np.argmax(detections[:, 4])
            best_det = detections[best_idx]
            class_id = int(best_det[5])

            # Return classification based on class_id
            if class_id == 0:
                return "Opened Eye"
            elif class_id == 1:
                return "Closed Eye"
            elif class_id == 2:
                return "Yawning"
            else:
                return "No Yawn"

        except Exception as e:
            print(f"Error in prediction: {e}")
            return "No Detection"

    def process_frames(self, frame):
        """Receives and stores the latest frame, then processes it for detection."""
        global num_of_blinks_gui, microsleep_duration_gui, num_of_yawns_gui
        global yawn_duration_gui, blinks_per_minute_gui, yawns_per_minute_gui

        # Store the latest frame globally inside the class
        self.current_frame = frame  

        try:
            self.eyes_state = self.predict()  # Predict using stored frame

            # Handle eye blink detection
            if self.eyes_state == "Closed Eye":
                if not self.eyes_still_closed:
                    self.eyes_still_closed = True
                    self.start = time.perf_counter()
                    self.num_of_blinks += 1
                    num_of_blinks_gui = self.num_of_blinks
                    self.current_blinks += 1
                self.microsleep_duration = time.perf_counter() - self.start
                microsleep_duration_gui = self.microsleep_duration
            else:
                self.eyes_still_closed = False
                self.microsleep_duration = 0
                microsleep_duration_gui = self.microsleep_duration

            # Handle Yawn Detection
            if self.eyes_state == "Yawning":
                if not self.yawn_in_progress:
                    self.start = time.perf_counter()
                    self.yawn_in_progress = True
                    self.yawn_duration = 0
                self.yawn_duration = time.perf_counter() - self.start
                yawn_duration_gui = self.yawn_duration

                if yawn_duration_gui > yawning_threshold and not self.yawn_finished:
                    self.yawn_finished = True
                    self.num_of_yawns += 1
                    num_of_yawns_gui = self.num_of_yawns
                    self.current_yawns += 1
                    print(f"Yawn detected! Total Yawns: {self.num_of_yawns}")

            else:
                if self.yawn_in_progress:
                    self.yawn_in_progress = False
                    self.yawn_finished = False

                self.yawn_duration = 0
                yawn_duration_gui = self.yawn_duration

        except Exception as e:
            print(f"Error in processing the frame: {e}")

    def update_blink_yawn_rate(self):
        """Updates blink and yawn rates every minute."""
        global blinks_per_minute_gui, yawns_per_minute_gui

        while not self.stop_event.is_set():
            time.sleep(self.time_window)  # Wait for 1 minute
            self.blinks_per_minute = self.current_blinks
            blinks_per_minute_gui = self.blinks_per_minute
            self.yawns_per_minute = self.current_yawns
            yawns_per_minute_gui = self.yawns_per_minute

            print(f"Updated Rates - Blinks: {self.blinks_per_minute} per min, Yawns: {self.yawns_per_minute} per min")

            # Reset current values for next cycle
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
        #    possible_fatigue_alert = 1
        #if blink_rate > 35 or yawning_rate > 5:
        #    highly_fatigued_alert = 1
        #elif blink_rate > 25 or yawning_rate > 3:
        #    possibly_fatigued_alert = 1

    def play_alert_sound(self):
        """Plays an alert sound for fatigue detection."""
        frequency = 1000
        duration = 500
        winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        """Runs the alert sound in a separate thread."""
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start() 