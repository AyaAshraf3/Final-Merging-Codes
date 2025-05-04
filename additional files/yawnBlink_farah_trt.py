import queue
import threading
import time
import cv2
import numpy as np
import sys
from thresholds import *
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Global Variables for GUI
num_of_blinks_gui = 0
microsleep_duration_gui = 0
num_of_yawns_gui = 0
yawn_duration_gui = 0
blinks_per_minute_gui = 0
yawns_per_minute_gui = 0

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

        # Initialize TensorRT model
        self.detect_drowsiness = TRTInference("drowsiness_model.engine")
        
        # Using Multi-Threading (Only for tracking blink/yawn rates)
        self.stop_event = threading.Event()
        self.blink_yawn_thread = threading.Thread(target=self.update_blink_yawn_rate)
        self.blink_yawn_thread.start()

    def predict(self):
        """Processes the current frame and returns the detected state using TensorRT."""
        if self.current_frame is None:
            return "No Detection"

        try:
            # Run inference using TensorRT
            output = self.detect_drowsiness.infer(self.current_frame)
            
            # Get class with highest confidence
            class_id = np.argmax(output[0])
            confidence = output[0][class_id]

            if confidence < 0.5:  # Confidence threshold
                return "No Detection"

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
            print(f"Error in TensorRT inference: {e}")
            return "No Detection"

    def process_frames(self, frame):
        """Receives and stores the latest frame, then processes it for detection."""
        global num_of_blinks_gui, microsleep_duration_gui, num_of_yawns_gui
        global yawn_duration_gui, blinks_per_minute_gui, yawns_per_minute_gui

        # Store the latest frame
        self.current_frame = frame  

        try:
            self.eyes_state = self.predict()

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
            time.sleep(self.time_window)
            self.blinks_per_minute = self.current_blinks
            blinks_per_minute_gui = self.blinks_per_minute
            self.yawns_per_minute = self.current_yawns
            yawns_per_minute_gui = self.yawns_per_minute

            print(f"Updated Rates - Blinks: {self.blinks_per_minute} per min, Yawns: {self.yawns_per_minute} per min")

            self.current_blinks = 0
            self.current_yawns = 0

    def fatigue_detection(self):
        """Triggers alerts based on fatigue detection using the latest frame."""
        if self.current_frame is None:
            return

        microsleep_duration = microsleep_duration_gui
        blink_rate = blinks_per_minute_gui
        yawning_rate = yawns_per_minute_gui

    def cleanup(self):
        """Cleanup resources when stopping the detector."""
        self.stop_event.set()
        if self.blink_yawn_thread.is_alive():
            self.blink_yawn_thread.join() 