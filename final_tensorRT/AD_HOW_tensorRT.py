import threading
import queue
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from collections import Counter
from PIL import Image
import sys
import json
from collections import deque
from pathlib import Path
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

torch.cuda.empty_cache()

# Global adjustable thresholds
CONF_THRESHOLD = 0.5  
IOU_THRESHOLD = 0.45   

# TensorRT setup
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
AD_ENGINE_PATH = r"D:\grad project\Driver-Monitoring-System\Activity_Detection\models_weights\activity_detection.engine"  # Update this path
HOW_ENGINE_PATH = r"D:\grad project\Driver-Monitoring-System\Hands_Off_Wheel\how_yolov7\weights\best_lastTrain.engine"  # Update this path

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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Global variables and parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = {
    0: "Safe driving",
    1: "Texting(right hand)",
    2: "Talking on the phone (right hand)",
    3: "Texting (left hand)",
    4: "Talking on the phone (left hand)",
    5: "Operating the radio",
    6: "Drinking",
    7: "Reaching behind",
    8: "Hair and makeup",
    9: "Talking to passenger(s)",
}

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load TensorRT engines
ad_engine = load_engine(AD_ENGINE_PATH)
how_engine = load_engine(HOW_ENGINE_PATH)

# Allocate buffers for both engines
ad_inputs, ad_outputs, ad_bindings, ad_stream = allocate_buffers(ad_engine)
how_inputs, how_outputs, how_bindings, how_stream = allocate_buffers(how_engine)

# Create execution contexts
ad_context = ad_engine.create_execution_context()
how_context = how_engine.create_execution_context()

# Get input dimensions for HOW engine
how_binding_shape = how_engine.get_binding_shape(0)
how_input_shape = (how_binding_shape[2], how_binding_shape[3])  # (height, width)
stride = 32  # Default stride for YOLOv7
img_size = (640, 640)  # Default input size for YOLOv7

# Queues for frames and results
frame_queue_AD = queue.Queue(maxsize=5)
frame_queue_HOW = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=100)

# Event to signal threads to stop
stop_event = threading.Event()

# Global status variables
per_frame_driver_activity = "Unknown (0.0%)"
per_frame_hands_on_wheel = "No (0.00)"
driver_state = "N/A"    # majority driver state
confidence_text = "N/A" # system alert
hands_state = "N/A"     # majority hands monitoring
hands_confidence = "N/A"  # majority hands monitoring confidence

# Global variable to hold the latest frame
latest_frame = None

def preprocess_HOW(frame):
    # Resize and normalize
    img = cv2.resize(frame, how_input_shape)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    return img

def predict_HOW(img, frame):
    try:
        # Copy input to device
        np.copyto(how_inputs[0][0], img.ravel())
        cuda.memcpy_htod_async(how_inputs[0][1], how_inputs[0][0], how_stream)

        # Execute inference
        how_context.execute_async_v2(bindings=how_bindings, stream_handle=how_stream.handle)

        # Copy output from device
        cuda.memcpy_dtoh_async(how_outputs[0][0], how_outputs[0][1], how_stream)
        how_stream.synchronize()

        # Process output
        output = how_outputs[0][0]
        output = output.reshape(1, -1, 6)  # Reshape to (batch, num_detections, 6)
        
        # Filter detections by confidence threshold
        detections = output[output[..., 4] > CONF_THRESHOLD]
        
        if len(detections) == 0:
            print("No hands-on-wheel detected, skipping frame.")
            return None, 0.0, None

        # Get the detection with highest confidence
        best_idx = np.argmax(detections[:, 4])
        best_det = detections[best_idx]
        
        # Extract box coordinates and confidence
        x1, y1, x2, y2 = best_det[:4]
        conf = best_det[4]
        cls = int(best_det[5])
        
        # Scale coordinates to original image size
        h, w = frame.shape[:2]
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        
        print(f"Detected HOW class: {cls}, confidence: {conf}")
        return cls, conf, [x1, y1, x2, y2]
        
    except Exception as e:
        print(f"Error in HOW prediction: {e}")
        return None, 0.0, None

def predict_activity_AD(frame):
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(Image.fromarray(img_rgb)).unsqueeze(0).numpy()

        # Copy input to device
        np.copyto(ad_inputs[0][0], img.ravel())
        cuda.memcpy_htod_async(ad_inputs[0][1], ad_inputs[0][0], ad_stream)

        # Execute inference
        ad_context.execute_async_v2(bindings=ad_bindings, stream_handle=ad_stream.handle)

        # Copy output from device
        cuda.memcpy_dtoh_async(ad_outputs[0][0], ad_outputs[0][1], ad_stream)
        ad_stream.synchronize()

        # Process output
        logits = ad_outputs[0][0]
        probabilities = softmax(logits)
        
        top_prob, top_index = np.max(probabilities), np.argmax(probabilities)
        top_label = class_labels[top_index]
        top_confidence = round(top_prob * 100, 2)
        
        return [(top_label, top_confidence)]
    except Exception as e:
        print(f"Error in activity prediction: {e}")
        return [("Unknown", 0.0)]

def capture_frames(video_path):
    cap = cv2.VideoCapture("http://127.0.0.1:5000/video_feed1")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 1 / 30  

    while not stop_event.is_set():
        ret, frame = cap.read()
        if frame is None:
            print("No frame available, waiting...")
            time.sleep(frame_time)
            continue
        if ret:
            if frame_queue_AD.full():
                frame_queue_AD.get()
            if frame_queue_HOW.full():
                frame_queue_HOW.get()
        frame_queue_AD.put(frame.copy())
        frame_queue_HOW.put(frame.copy())
        print(f"Added frame to queues - AD size: {frame_queue_AD.qsize()}, HOW size: {frame_queue_HOW.qsize()}")
        time.sleep(frame_time)

def process_frames_HOW():
    while not stop_event.is_set():
        try:
            frame = frame_queue_HOW.get()
            img = preprocess_HOW(frame)
            highest_cls, highest_conf, best_box = predict_HOW(img, frame)
            if result_queue.qsize() < 100:
                result_queue.put(("HOW", frame, (highest_cls, highest_conf, best_box)))
            else:
                time.sleep(0.1)
                print("Processing batch of 100 frames (HOW)")
                while not result_queue.empty():
                    result_queue.get()
        except Exception as e:
            print(f"Error in HOW: {e}")

def process_frames_AD():
    while not stop_event.is_set():
        try:
            frame = frame_queue_AD.get()
            print("Processing AD frame...")
            top_prediction = predict_activity_AD(frame)
            print(f"AD prediction: {top_prediction}")
            if result_queue.qsize() < 100:
                result_queue.put(("AD", frame, top_prediction))
            else:
                time.sleep(0.1)
                print("Processing batch of 100 frames (AD)")
                while not result_queue.empty():
                    result_queue.get()
            print(f"AD prediction added to queue, new size: {result_queue.qsize()}")
        except Exception as e:
            print(f"Error in activity prediction: {e}")

def majority_how_update():
    global hands_state, hands_confidence
    queue_list = list(result_queue.queue)  
    how_predictions = [predictions for source, _, predictions in queue_list if source == "HOW"]

    print(f"Current queue size: {result_queue.qsize()}, HOW frame count: {len(how_predictions)}")

    if (result_queue.qsize()) < 25:  
        print(f"Queue size is {result_queue.qsize()}, waiting for 50 HOW frames...")
        return  

    hands_on_counter = 0
    hands_off_counter = 0
    for predictions in how_predictions:
        detected_label = predictions[0]
        if detected_label == 1:
            hands_on_counter += 1
        else:
            hands_off_counter += 1
    if hands_on_counter > hands_off_counter:
        hands_state = "Hands On Wheel"
        hands_confidence = "✅Driver is in control"
    else:
        hands_state = "Hands Off Wheel"
        hands_confidence = "⚠🚨WARNING! Hands off wheel detected!"
    print(f"HANDs Majority Updated: {hands_state}, {hands_confidence}")

def majority_class_update():
    global driver_state, confidence_text
    queue_list = list(result_queue.queue)
    ad_predictions = [predictions for source, _, predictions in queue_list if source == "AD"]
    
    print(f"Current queue size: {result_queue.qsize()}, AD frame count: {len(ad_predictions)}")

    if (result_queue.qsize()) < 100:  
        print(f"Queue size is {result_queue.qsize()}, waiting for 100 AD frames...")
        return 
    safe_counter = 0
    unsafe_counter = 0
    for predictions in ad_predictions:
        driver_label, _ = predictions[0]
        if driver_label == "Safe driving":
            safe_counter += 1
        else:
            unsafe_counter += 1
    if safe_counter > unsafe_counter:
        driver_state = "✅Safe driving"
        confidence_text = "Good job,you're driving safely"
    else:
        driver_state = "❌Unsafe driving"
        confidence_text = "⚠🚨ALERT!!! PAY ATTENTION TO THE ROAD"
    if hands_state == "Hands Off Wheel":
        driver_state = "❌Unsafe driving"
        confidence_text = "⚠🚨ALERT!!! PUT YOUR HANDS ON THE WHEEL"
    print(f"AD Majority Updated: {driver_state}, {confidence_text}")

def update_status_loop():
    global per_frame_driver_activity, per_frame_hands_on_wheel, latest_frame
    while not stop_event.is_set():
        try:
            queue_list = list(result_queue.queue)
            driver_state_gui = "Unknown"
            conf_gui = "N/A"
            highest_cls = "N/A"
            highest_conf = 0.0
            for source, frame, prediction in queue_list[-2:]:
                if frame is not None:
                    latest_frame = frame
                if source == "AD":
                    driver_state_gui, conf_gui = prediction[0]
                elif source == "HOW":
                    highest_cls, highest_conf, best_box = prediction

            per_frame_driver_activity = f"{driver_state_gui} ({conf_gui}%)"
            yes_no = "Yes" if highest_cls == 1 else "No"
            per_frame_hands_on_wheel = f"{yes_no} ({highest_conf:.2f})"
            majority_class_update()
            majority_how_update()

            data = {
                "per_frame_driver_activity": per_frame_driver_activity,
                "per_frame_hands_on_wheel": per_frame_hands_on_wheel,
                "majority_driver_state": driver_state,
                "system_alert": confidence_text,
                "hands_monitoring": hands_state,
                "hands_monitoring_confidence": hands_confidence
            }
            with open("status.json", "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error in update_status_loop: {e}")
        time.sleep(0.1)

if __name__ == "__main__":
    video_path = input("Enter the video file path (or press Enter to use the live feed URL): ")
    if not video_path:
        print("No video file provided. Please enter a valid video file path.")
        sys.exit()
    else:
        print(f"Using video file: {video_path}")

    start_time = time.time()

    # Start threads for capture, processing, and status update
    capture_thread = threading.Thread(target=capture_frames, args=(video_path,))
    ad_thread = threading.Thread(target=process_frames_AD)
    how_thread = threading.Thread(target=process_frames_HOW)
    status_thread = threading.Thread(target=update_status_loop)

    capture_thread.start()
    ad_thread.start()
    how_thread.start()
    status_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupt received, stopping threads...")
        stop_event.set()
        capture_thread.join()
        ad_thread.join()
        how_thread.join()
        status_thread.join()

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    sys.exit(0)
