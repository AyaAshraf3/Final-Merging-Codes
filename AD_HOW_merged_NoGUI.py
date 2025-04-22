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

torch.cuda.empty_cache()

YOLOV7_REPO_PATH = r"D:\grad project\Driver-Monitoring-System\final merging codes with live streaming\yolov7"
sys.path.append(YOLOV7_REPO_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.datasets import letterbox

# Global adjustable thresholds
CONF_THRESHOLD = 0.5  
IOU_THRESHOLD = 0.45   

# Define Model for Activity Detection
class CustomModel(nn.Module):
    def __init__(self, model_path, labels, classes, device="CPU"):
        super(CustomModel, self).__init__()
        self.device = device
        self.model = self.load_model(model_path, len(classes))
        self.model.eval()
        self.labels = labels
        self.classes = classes
        self.model = self.model.to(self.device)

    def load_model(self, model_path, num_classes):
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        return model

# Global variables and parameters (including those sent to the JSON file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\grad project\Driver-Monitoring-System\Activity_Detection\models_weights\fine_tuned_mobilenetv3_with_aug.pth"
labels = list(range(0, 10))
class_labels  = {
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

weights_path = r"D:\grad project\Driver-Monitoring-System\Hands_Off_Wheel\how_yolov7\weights\best_lastTrain.pt"
model_HOW = attempt_load(weights_path, map_location=device)
print(f"Using device: {device}")
stride = int(model_HOW.stride.max())  
img_size = check_img_size(640, s=stride)  

# Queues for frames and results
frame_queue_AD = queue.Queue(maxsize=5)
frame_queue_HOW = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=100)

# Event to signal threads to stop
stop_event = threading.Event()

# Global status variables (matching JSON keys)
per_frame_driver_activity = "Unknown (0.0%)"
per_frame_hands_on_wheel = "No (0.00)"
driver_state = "N/A"    # majority driver state
confidence_text = "N/A" # system alert
hands_state = "N/A"     # majority hands monitoring
hands_confidence = "N/A"  # majority hands monitoring confidence

# Global variable to hold the latest frame (for web display)
latest_frame = None

def preprocess_HOW(frame):
    img = letterbox(frame, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

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
            highest_cls, highest_conf, best_box = predict_HOW(img, model_HOW, frame)
            if result_queue.qsize() < 100:
                result_queue.put(("HOW", frame, (highest_cls, highest_conf, best_box)))
            else:
                time.sleep(0.1)
                print("Processing batch of 100 frames (HOW)")
                while not result_queue.empty():
                    result_queue.get()
        except Exception as e:
            print(f"Error in HOW: {e}")

def predict_HOW(img, model, frame):
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD, agnostic=False)
    highest_conf = 0
    highest_cls = None
    best_box = None
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                cls = int(cls)
                if conf > highest_conf:
                    highest_conf = conf.item()
                    highest_cls = cls
                    best_box = [x.item() for x in xyxy]
    print(f"Detected HOW class: {highest_cls}, confidence: {highest_conf}")
    if highest_cls is None:
        print("No hands-on-wheel detected, skipping frame.")
        return None, 0.0, None
    return highest_cls, highest_conf, best_box

def process_frames_AD():
    model = CustomModel(model_path, labels, class_labels, device=device).model
    while not stop_event.is_set():
        try:
            frame = frame_queue_AD.get()
            print("Processing AD frame...")
            top_prediction = predict_activity_AD(frame, model)
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

def predict_activity_AD(frame, model):
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_index = torch.topk(probabilities, 1)
            top_label = [class_labels[idx.item()] for idx in top_index[0]]
            top_confidence = [round(prob.item() * 100, 2) for prob in top_prob[0]]
            return list(zip(top_label, top_confidence))
    except Exception as e:
        print(f"Error in activity prediction: {e}")
        return [("Unknown", 0.0)]

def majority_how_update():
    global hands_state, hands_confidence
    # Convert queue to list before filtering
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
    # Ensure at least 100 frames related to AD are collected
    queue_list = list(result_queue.queue)  # Convert queue to list
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
    """
    This loop periodically retrieves the latest predictions from the result queue,
    updates per-frame parameters, computes majority statuses, writes the results to 'status.json',
    and also saves the most recent frame to the global variable 'latest_frame' (for the Flask video stream).
    """
    global per_frame_driver_activity, per_frame_hands_on_wheel, latest_frame
    while not stop_event.is_set():
        try:
            queue_list = list(result_queue.queue)
            driver_state_gui = "Unknown"
            conf_gui = "N/A"
            highest_cls = "N/A"
            highest_conf = 0.0
            # Look at the most recent predictions (if available)
            for source, frame, prediction in queue_list[-2:]:
                if frame is not None:
                    latest_frame = frame  # update the global frame for web display
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

# # ===== Temporary Flask Web Interface =====
# # You can remove this function later if not needed.
# from flask import Flask, Response, jsonify

# def start_flask_server():
#     app = Flask(__name__)

#     @app.route('/')
#     def index():
#         # A simple HTML page that displays the video stream and the system status.
#         html = '''
#         <html>
#         <head>
#             <title>Driver Monitoring System</title>
#         </head>
#         <body>
#             <h1>Driver Monitoring System</h1>
#             <img src="/video_feed" width="640" height="480">
#             <h2>Status</h2>
#             <div id="status"></div>
#             <script>
#             function fetchStatus(){
#               fetch('/status').then(response => response.json()).then(data => {
#                 document.getElementById('status').innerHTML = 
#                   '<p>Per Frame Driver Activity: ' + data.per_frame_driver_activity + '</p>' +
#                   '<p>Per Frame Hands on Wheel: ' + data.per_frame_hands_on_wheel + '</p>' +
#                   '<p>Majority Driver State: ' + data.majority_driver_state + '</p>' +
#                   '<p>System Alert: ' + data.system_alert + '</p>' +
#                   '<p>Hands Monitoring: ' + data.hands_monitoring + '</p>' +
#                   '<p>Hands Monitoring Confidence: ' + data.hands_monitoring_confidence + '</p>';
#               });
#             }
#             setInterval(fetchStatus, 1000);
#             </script>
#         </body>
#         </html>
#         '''
#         return html

#     def gen():
#         """Video streaming generator function."""
#         global latest_frame
#         while True:
#             if latest_frame is None:
#                 time.sleep(0.1)
#                 continue
#             ret, jpeg = cv2.imencode('.jpg', latest_frame)
#             if not ret:
#                 continue
#             frame_bytes = jpeg.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#             time.sleep(0.1)

#     @app.route('/video_feed')
#     def video_feed():
#         return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

#     @app.route('/status')
#     def get_status():
#         try:
#             with open("status.json") as f:
#                 data = json.load(f)
#             return jsonify(data)
#         except Exception as e:
#             return jsonify({"error": str(e)})

#     # Start Flask (you may change the host/port as needed)
#     app.run(host='0.0.0.0', port=5000)

# # ===== End of Flask Web Interface =====

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

    # # Start the Flask server in a separate thread for testing the web interface.
    # flask_thread = threading.Thread(target=start_flask_server)
    # flask_thread.daemon = True
    # flask_thread.start()

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
