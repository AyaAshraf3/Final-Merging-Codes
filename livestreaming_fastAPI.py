from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import threading
import time
import json
import os
import numpy as np

app = FastAPI()

# ─────────────────────────────────────
# Multi-threaded Camera Capture Class
# ─────────────────────────────────────
class CameraStream:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.running = True
        self.last_frame_time = time.time()
        self.error_count = 0
        self.MAX_ERRORS = 5
        self.initialize_camera()
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def initialize_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
                time.sleep(0.5)  # Give time for camera to properly release
            
            # Try different backend APIs
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.camera_id + cv2.CAP_DSHOW)
                    if self.cap.isOpened():
                        break
                    self.cap.release()
                except Exception:
                    continue

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
            
            # Read a test frame
            success, _ = self.cap.read()
            if not success:
                raise RuntimeError("Failed to read test frame")

            print(f"✅ Camera {self.camera_id} initialized successfully")
            self.error_count = 0
            return True
        except Exception as e:
            print(f"⚠️ Error initializing camera {self.camera_id}: {str(e)}")
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            return False

    def update_frames(self):
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if self.error_count < self.MAX_ERRORS:
                        self.error_count += 1
                        print(f"⚠️ Camera {self.camera_id} lost connection, attempting to reinitialize ({self.error_count}/{self.MAX_ERRORS})")
                        if self.initialize_camera():
                            continue
                    time.sleep(1)
                    continue

                success, frame = self.cap.read()
                if success:
                    self.frame = frame
                    self.last_frame_time = time.time()
                    self.error_count = 0
                else:
                    current_time = time.time()
                    if current_time - self.last_frame_time > 3.0:  # No frames for 3 seconds
                        print(f"⚠️ No frames from camera {self.camera_id} for 3 seconds, reinitializing...")
                        self.initialize_camera()
                time.sleep(0.02)  # Prevent CPU overuse

            except Exception as e:
                print(f"⚠️ Error in camera {self.camera_id} thread: {str(e)}")
                self.error_count += 1
                if self.error_count >= self.MAX_ERRORS:
                    print(f"❌ Too many errors for camera {self.camera_id}, stopping attempts")
                    break
                time.sleep(1)

    def get_frame(self):
        if self.frame is None or time.time() - self.last_frame_time > 5.0:
            return None
        return self.frame.copy()  # Return a copy to prevent race conditions

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        self.running = False
        self.thread.join()
        if self.cap is not None:
            self.cap.release()

# Initialize both camera feeds with correct indices
print("Initializing cameras...")
camera1 = CameraStream(0)  # Try primary camera first
camera2 = CameraStream(1)  # Try secondary camera

def generate_frames(camera_stream, name='cam'):
    while True:
        try:
            frame = camera_stream.get_frame()
            if frame is None:
                # Return a blank frame if camera is not working
                frame = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Disconnected", (120, 180), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Error in generate_frames for {name}: {str(e)}")
            time.sleep(0.1)

@app.get("/video_feed1")
def video_feed1():
    return StreamingResponse(
        generate_frames(camera1, 'cam1'),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video_feed2")
def video_feed2():
    return StreamingResponse(
        generate_frames(camera2, 'cam2'),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
      <head>
        <title>Driver Monitoring Live Stream</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        
        <style>
          html, body {
            margin: 0;
            padding: 0;
          }

          body {
            min-height: 100vh;
            font-family: Arial, sans-serif;
            background: linear-gradient(to bottom, #7D33A3, #6882C5);
            background-repeat: no-repeat;
            background-size: cover;
          }

          .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            align-items: flex-start;
            gap: 40px;
            padding: 20px;
          }

          .card {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            padding: 20px;
            width: 100%;
            max-width: 700px;
            box-sizing: border-box;
          }

          .card h2 {
            margin-top: 0;
          }

          .camera-container {
            position: relative;
            width: 100%;
            max-width: 640px;
            margin: 0 auto;
          }

          .camera-container img {
            display: block;
            width: 100%;
            height: auto;
          }

          .status-panel {
            margin-top: 20px;
            border: 1px solid #ccc;
            padding: 10px;
          }

          .confidence-box {
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
          }

          .confidence-good {
            background-color: rgba(0, 255, 0, 0.2);
            color: green;
          }

          .confidence-bad {
            background-color: rgba(255, 0, 0, 0.2);
            color: red;
          }
        </style>
        
        <script>
          function updateStatus() {
            fetch('/status')
              .then(response => response.json())
              .then(data => {
                // Update Camera 1 status panel
                document.getElementById("statusInfo").innerHTML = `
                  <div>
                    <strong style="font-size: 18px;">Activity and hands detection</strong>
                    <hr>
                    <div class="confidence-box ${data.camera1.system_alert === "Good job,you're driving safely" ? 'confidence-good' : 'confidence-bad'}">
                      System Alert: ${data.camera1.system_alert}
                    </div>
                    <h3 style="margin: 0;">Per Frame Prediction</h3>
                    <p style="margin: 0;">
                      <b>Driver Activity:</b> ${data.camera1.per_frame_driver_activity}<br>
                      <b>Hands-on-Wheel:</b> ${data.camera1.per_frame_hands_on_wheel}
                    </p>
                    <hr>
                    <h3 style="margin: 0;">State Monitoring</h3>
                    <p style="margin: 0;">
                      <b>Majority Driver State:</b> ${data.camera1.majority_driver_state}<br>
                      <b>System Alert:</b> ${data.camera1.system_alert}<br>
                      <b>Hands Monitoring:</b> ${data.camera1.hands_monitoring}<br>
                      <b>Hands Monitoring Confidence:</b> ${data.camera1.hands_monitoring_confidence}
                    </p>
                  </div>
                `;
                
                // Update Camera 2 status panel
                document.getElementById("statusInfo2").innerHTML = `
                  <div>
                    <strong style="font-size: 18px;">Fatigue detection</strong>
                    <hr>
                    <h3 style="margin: 0;">Gaze Detection</h3>
                    <p style="margin: 0;">
                      <b>Position:</b> ${data.camera2.gaze_center}<br>
                      <b>Status:</b> ${data.camera2.gaze_status}
                    </p>
                    <hr>
                    <h3 style="margin: 0;">Head Movement</h3>
                    <p style="margin: 0;">
                      <b>Pitch:</b> ${data.camera2.pitch}<br>
                      <b>Yaw:</b> ${data.camera2.yaw}<br>
                      <b>Roll:</b> ${data.camera2.roll}<br>
                      <b>Head Status:</b> ${data.camera2.head_status}
                    </p>
                    <hr>
                    <h3 style="margin: 0;">Distraction</h3>
                    <p style="margin: 0;">
                      ${data.camera2.distraction}
                    </p>
                    <hr>
                    <h3 style="margin: 0;">Drowsiness Detection</h3>
                    <p style="margin: 0;">
                      <b>Blinks:</b> ${data.camera2.blinks}<br>
                      <b>Microsleep Duration:</b> ${data.camera2.microsleep_duration}<br>
                      <b>Yawns:</b> ${data.camera2.yawns}<br>
                      <b>Yawn Duration:</b> ${data.camera2.yawn_duration}<br>
                      <b>Blinks Per Minute:</b> ${data.camera2.blinks_per_minute}<br>
                      <b>Yawns Per Minute:</b> ${data.camera2.yawns_per_minute}<br>
                      <b>Alert:</b> ${data.camera2.alert}
                    </p>
                  </div>
                `;
              })
              .catch(err => {
                console.error("Error fetching status:", err);
              });
          }

          // Update status every second
          window.onload = function() {
            updateStatus();
            setInterval(updateStatus, 1000);
          };
        </script>
      </head>

      <body>
        <div class="container">
          <!-- Camera 1 Card -->
          <div class="card">
            <h2>Camera 1 Feed</h2>
            <div class="camera-container">
              <img src="/video_feed1" alt="Camera 1 Feed"/>
            </div>
            <div id="statusInfo" class="status-panel">
              <!-- status panel 1 updates here -->
            </div>
          </div>

          <!-- Camera 2 Card -->
          <div class="card">
            <h2>Camera 2 Feed</h2>
            <div class="camera-container">
              <img src="/video_feed2" alt="Camera 2 Feed"/>
            </div>
            <div id="statusInfo2" class="status-panel">
              <!-- status panel 2 updates here -->
            </div>
          </div>
        </div>
      </body>
    </html>
    """

@app.get("/status")
async def status():
    file1 = "status.json"
    file2 = "status_driver_fatigue.json"
    
    # Read Camera 1 data
    if os.path.exists(file1):
        try:
            with open(file1, "r") as f:
                data1 = json.load(f)
        except Exception as e:
            print("Error reading status.json:", e, flush=True)
            data1 = {
                "per_frame_driver_activity": "Error reading file",
                "per_frame_hands_on_wheel": "N/A",
                "majority_driver_state": "N/A",
                "system_alert": "N/A",
                "hands_monitoring": "N/A",
                "hands_monitoring_confidence": "N/A"
            }
    else:
        data1 = {
            "per_frame_driver_activity": "No data yet",
            "per_frame_hands_on_wheel": "No data yet",
            "majority_driver_state": "No data yet",
            "system_alert": "No data yet",
            "hands_monitoring": "No data yet",
            "hands_monitoring_confidence": "No data yet"
        }
    
    # Read Camera 2 data
    if os.path.exists(file2):
        try:
            with open(file2, "r") as f:
                data2 = json.load(f)
        except Exception as e:
            print("Error reading status_driver_fatigue.json:", e, flush=True)
            data2 = {
                "gaze_center": "Error reading file",
                "gaze_status": "N/A",
                "pitch": "N/A",
                "yaw": "N/A",
                "roll": "N/A",
                "head_status": "N/A",
                "distraction": "N/A",
                "blinks": "N/A",
                "microsleep_duration": "N/A",
                "yawns": "N/A",
                "yawn_duration": "N/A",
                "blinks_per_minute": "N/A",
                "yawns_per_minute": "N/A",
                "alert": "N/A"
            }
    else:
        data2 = {
            "gaze_center": "No data yet",
            "gaze_status": "No data yet",
            "pitch": "No data yet",
            "yaw": "No data yet",
            "roll": "No data yet",
            "head_status": "No data yet",
            "distraction": "No data yet",
            "blinks": "No data yet",
            "microsleep_duration": "No data yet",
            "yawns": "No data yet",
            "yawn_duration": "No data yet",
            "blinks_per_minute": "No data yet",
            "yawns_per_minute": "No data yet",
            "alert": "No data yet"
        }
    
    return {"camera1": data1, "camera2": data2}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("livestreaming_fastAPI:app", host="0.0.0.0", port=5000)
