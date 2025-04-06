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
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import sys
import time
from collections import deque
from pathlib import Path
#from webRTC_livestream import video_track, start_webrtc_loop
import json
#from shared_cameras import camera1 # Import shared camera instance


torch.cuda.empty_cache()

YOLOV7_REPO_PATH = r"D:\grad project\Merging_codes_live_streaming\yolov7"
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

# Global Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\grad project\imgClass_AD\activity detection models\fine_tuned_mobilenetv3_with_aug_AD.pth"
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

weights_path = r"D:\grad project\imgClass_AD\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\best_lastTrain.pt"
model_HOW = attempt_load(weights_path, map_location=device)
print(f"Using device: {device}")
stride = int(model_HOW.stride.max())  
img_size = check_img_size(640, s=stride)  

# ✅ Replace multiprocessing with queue.Queue()
frame_queue_AD = queue.Queue(maxsize=5)
frame_queue_HOW = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=100)

# ✅ Use threading Event
stop_event = threading.Event()

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

    # Use camera1's frame instead of a video file
    while not stop_event.is_set():
        ret, frame = cap.read()
        # frame = camera1.get_frame()
        if frame is None:
            print("No frame available from camera1, waiting...")
            time.sleep(frame_time)
            continue
        if ret:
            if frame_queue_AD.full():
                frame_queue_AD.get()
            if frame_queue_HOW.full():
                frame_queue_HOW.get()
            
        frame_queue_AD.put(frame.copy())  
        frame_queue_HOW.put(frame.copy())
        print(f"Added frame to queues - AD size: {frame_queue_AD.qsize()}, HOW size: {frame_queue_HOW.qsize()}")  # Debuggin
        time.sleep(frame_time)  

def process_frames_HOW():
    while not stop_event.is_set():
        try:
            frame = frame_queue_HOW.get()
            img = preprocess_HOW(frame)
            highest_cls, highest_conf, best_box = predict_HOW(img, model_HOW, frame)

            # Ensure queue doesn't overflow
            if result_queue.qsize() < 100:
                result_queue.put(("HOW", frame, (highest_cls, highest_conf, best_box)))
            # If queue reaches 100 frames, process and reset
            else:
                time.sleep(0.1)  # Simulate processing time
                print("Processing batch of 100 frames")
                # You can perform additional processing here if needed
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
    
    print(f"Detected class: {highest_cls}, confidence: {highest_conf}")
     # **Add this check to print debug message**
    if highest_cls is None:
        print("No hands-on-wheel detected, skipping frame.")
        return None, 0.0, None  # Return empty values if no detection

    return highest_cls, highest_conf, best_box

def process_frames_AD():
    model = CustomModel(model_path, labels, class_labels, device=device).model

    while not stop_event.is_set():
        try:
            frame = frame_queue_AD.get()
            print("Processing AD frame...")  # Debugging
            top_prediction = predict_activity_AD(frame, model)
            print(f"AD prediction: {top_prediction}")  # Debugging
            if result_queue.qsize() < 100:
                result_queue.put(("AD", frame, top_prediction))
            # If queue reaches 100 frames, process and reset
            else:
                time.sleep(0.1)  # Simulate processing time
                print("Processing batch of 100 frames")
                # You can perform additional processing here if needed
                while not result_queue.empty():
                    result_queue.get()

            print(f"AD prediction added to queue, new size: {result_queue.qsize()}")  # Debugging
        except Exception as e:
            print(f"Error in activity prediction: {e}")
            
def predict_activity_AD(frame, model):
    """Predict driver activity using ResNet-18."""
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
        return "Unknown"

class ActivityDetection(QMainWindow):
    def __init__(self, result_queue, stop_event, video_path):
        super().__init__()

        # self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

        self.alert_text = ""
        
        # variables for JSON file
        self.per_frame_driver_activity = "Unknown (0.0%)"
        self.per_frame_hands_on_wheel = "No (0.00)"
        self.driver_state = "N/A"  # majority driver state
        self.confidence = "N/A"    # system alert
        self.hands_state = "N/A"   # majority hands monitoring
        self.hands_confidence = "N/A"  # majority hands monitoring confidence

        self.setWindowTitle("Driver Activity Detection")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: white;")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 2px solid black;")
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 18px;")
        self.layout.addWidget(self.info_label)

        self.update_info()

        self.timer = self.startTimer(10)  # 10ms update interval

    def update_info(self):
        """Update activity info on UI."""
        info_text = (
            f"<div style='font-family: Arial, sans-serif; color: #333; font-size: 18px;'>"
            f"<h2 style='text-align: center; color: #4CAF50; font-size: 24px;'>Driver Activity Detection</h2>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"{self.alert_text}"
            f"<p style='font-size: 20px;'><b> Current Driver State:</b> {self.driver_state}</p>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"</div>"
        )
        self.info_label.setText(info_text)

    def timerEvent(self, event):
        """Update UI with processed frame results."""
        try:
            if not self.result_queue:
                print("Timer Event: No results in queue.")
                return
            
            print(f"Timer Event: Processing queue with size {result_queue.qsize()}")  # Debugging
            # Extract predictions
            frame = None
            AD_prediction = None
            HOW_prediction = None
            driver_state_gui = "Unknown"
            conf_gui = "N/A"
            highest_cls = "N/A"
            highest_conf = 0.0

            # Retrieve the latest two predictions
            queue_list = list(result_queue.queue)  # Convert queue to list before iteration

            # Retrieve the latest two predictions
            for source, frame, prediction in queue_list[-2:]:  
                if source == "AD":
                    AD_prediction = prediction
                elif source == "HOW":
                    HOW_prediction = prediction
            
            # Display frame with both predictions
            if frame is not None:
                self.display_frame(frame)
            
            # Extract AD info
            if AD_prediction:
                driver_state_gui, conf_gui = AD_prediction[0]

            # Extract HOW info
            if HOW_prediction:
                highest_cls, highest_conf, best_box = HOW_prediction
                color = (0, 255, 0) if highest_cls == 1 else (0, 0, 255)
                if best_box:
                    cv2.rectangle(frame, (int(best_box[0]), int(best_box[1])), 
                                        (int(best_box[2]), int(best_box[3])), color, 2)
                    cv2.putText(frame, f"{highest_cls} {highest_conf:.2f}",
                                (int(best_box[0]), int(best_box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # **Call Majority_Class() and Majority_HOW() to process accumulated frames**
            self.Majority_Class()
            self.Majority_HOW()
            
            # Then build the “per frame” text:

            self.per_frame_driver_activity = f"{driver_state_gui} ({conf_gui}%)"
            yes_no = "Yes" if highest_cls == 1 else "No"
            self.per_frame_hands_on_wheel = f"{yes_no} ({highest_conf:.2f})"

            # **Update GUI text with majority class results**
            info_text = (
                f"<div style='font-family: Arial, sans-serif; color: #333; font-size: 18px;'>"
                f"<h2 style='text-align: center; color: #4CAF50; font-size: 24px;'>Driver Monitoring System</h2>"
                f"<hr style='border: 1px solid #4CAF50;'>"
                
                # **Per Frame Prediction section**
                f"<h3 style='color: #4CAF50;'>Per Frame Prediction</h3>"
                f"<b>Driver Activity:</b> {driver_state_gui} ({conf_gui}%)<br>"
                f"<b>Hands-on-Wheel:</b> {'Yes' if highest_cls == 1 else 'No'} ({highest_conf:.2f})<br>"
                f"<hr style='border: 1px solid #4CAF50;'>"

                # **Majority Class State Monitoring**
                f"<h3 style='color: #4CAF50;'>State Monitoring</h3>"
                f"<b>Majority Driver State:</b> {self.driver_state}<br>"
                f"<b>System Alert:</b> {self.confidence}<br>"
                f"<b>Hands Monitoring:</b> {self.hands_state}<br>"
                f"<b>Hands Monitoring Confidence:</b> {self.hands_confidence}<br>"
                
                f"</div>"
                )

            self.info_label.setText(info_text)
            
            #Write to status.json:
            self.write_camera_ONE_status_json()

        except Exception as e:
            print(f"Error in timerEvent: {e}")

    def write_camera_ONE_status_json(self):
        # Gather the exact 6 variables you want to display in Flask:
        data = {
            "per_frame_driver_activity": self.per_frame_driver_activity,  # e.g. "Safe driving (100.0%)"
            "per_frame_hands_on_wheel": self.per_frame_hands_on_wheel,    # e.g. "Yes (0.86)"
            "majority_driver_state": self.driver_state,                   # e.g. "✅Safe driving"
            "system_alert": self.confidence,                              # e.g. "Good boy"
            "hands_monitoring": self.hands_state,                         # e.g. "Hands On Wheel"
            "hands_monitoring_confidence": self.hands_confidence          # e.g. "✅Driver is in control"
        }
        try:
            with open("status.json", "w") as f:
                json.dump(data, f)
        except Exception as e:
            print("Error writing status.json:", e)



    def display_frame(self, frame):
        """Display frame in PyQt5 GUI."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))
        #video_track.update_frame(frame)

        
        
    def Majority_HOW(self):
        """Process majority class only when exactly 100 HOW frames are available."""
        print("Majority_HOW entered")

        # Convert queue to list before filtering
        queue_list = list(result_queue.queue)  
        how_predictions = [predictions for source, _, predictions in queue_list if source == "HOW"]

        print(f"Current queue size: {result_queue.qsize()}, HOW frame count: {len(how_predictions)}")

        if (result_queue.qsize()) < 25:  
            print(f"Queue size is {result_queue.qsize()}, waiting for 50 HOW frames...")
            return  

        try:
            hands_on_counter = 0
            hands_off_counter = 0
            
            # **Count only HOW-related predictions**
            for predictions in how_predictions:
                detected_label = predictions[0]   # Extract HOW label
                if detected_label == 1:  # Assuming class 1 means hands ON wheel
                    hands_on_counter += 1   
                else:
                    hands_off_counter += 1
                
            # **Determine majority class**
            if hands_on_counter > hands_off_counter:
                self.hands_state = "Hands On Wheel"
                self.hands_confidence = "✅Driver is in control"
            else:
                self.hands_state = "Hands Off Wheel"
                self.hands_confidence = "⚠🚨WARNING! Hands off wheel detected!"

            print(f"HANDS Monitoring Updated: {self.hands_state}, {self.hands_confidence}")  # ✅ Debugging
            print(f"hands_on_counter: {hands_on_counter}, hands_off_counter: {hands_off_counter}")

            # ✅ Clear HOW-related frames after processing
            # how_predictions.clear()
            # while not result_queue.empty():
            #     source, _, _ = result_queue.queue[0]  # Peek at first item
            #     if source == "HOW":
            #         result_queue.get()  # Remove only HOW-related frames
            #     else:
            #         break  # Stop when non-HOW frames are reached

        except Exception as e:
            print(f"Error in Majority_HOW: {e}")


    def Majority_Class(self):
        """Process majority class only when exactly 100 AD frames are available."""
        print("Majority_Class entered")

        # Ensure at least 100 frames related to AD are collected
        queue_list = list(result_queue.queue)  # Convert queue to list
        ad_predictions = [predictions for source, _, predictions in queue_list if source == "AD"]
        
        print(f"Current queue size: {result_queue.qsize()}, AD frame count: {len(ad_predictions)}")
    
        if (result_queue.qsize()) < 100:  
            print(f"Queue size is {result_queue.qsize()}, waiting for 100 AD frames...")
            return  

        try:
            safe_counter = 0
            unsafe_counter = 0
            
            # **Count only AD-related predictions**
            for predictions in ad_predictions:
                driver_label, _ = predictions[0]  # Extract AD label
                if driver_label == "Safe driving":
                    safe_counter += 1   
                else:
                    unsafe_counter += 1
            
            # **Determine majority class**
            if safe_counter > unsafe_counter:
                self.driver_state = "✅Safe driving"
                self.confidence = "Good boy"
            else:
                self.driver_state = "❌Unsafe driving"
                self.confidence = "⚠🚨ALERT!!! PAY ATTENTION TO THE ROAD"
            
            if self.hands_state == "Hands Off Wheel":
                self.driver_state = "❌Unsafe driving"
                self.confidence = "⚠🚨ALERT!!! PUT YOUR HANDS ON THE WHEEL"
            
            print(f"State Monitoring Updated: {self.driver_state}, {self.confidence}")  # ✅ Debugging
            print(f"safe_counter: {safe_counter}, unsafe_counter: {unsafe_counter}")
            # **Reset only the AD-related part of the queue**
            #self.result_queue[:] = [entry for entry in self.result_queue if entry[0] != "AD"]

        except Exception as e:
            print(f"Error in Majority_Class: {e}")

if __name__ == "__main__":
    #start_webrtc_loop()
    app = QApplication(sys.argv)

    video_path, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
    if not video_path:
        print("No video selected. Exiting.")
        sys.exit()

    start_time = time.time()

    # ✅ Replace multiprocessing with threading
    capture_thread = threading.Thread(target=capture_frames, args=(video_path,))
    ad_thread = threading.Thread(target=process_frames_AD)
    how_thread = threading.Thread(target=process_frames_HOW)

    capture_thread.start()
    ad_thread.start()
    how_thread.start()

    window = ActivityDetection(result_queue, stop_event, video_path)
    window.show()
    exit_code = app.exec_()

    # ✅ Ensure threads exit properly
    stop_event.set()
    capture_thread.join()
    ad_thread.join()
    how_thread.join()
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    sys.exit(exit_code)
