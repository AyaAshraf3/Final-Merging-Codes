from flask import Flask, Response, render_template, jsonify
from waitress import serve
import threading
import cv2
import time
import json
import os


app = Flask(__name__)

# ─────────────────────────────────────
# Multi-threaded Camera Capture Class
# ─────────────────────────────────────
class CameraStream:
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                self.frame = frame
            time.sleep(0.01)

    def get_frame(self):
        return self.frame
      
    def isOpened(self):
        return self.cap.isOpened()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# ─────────────────────────────
# Initialize both camera feeds
# ─────────────────────────────
camera1 = CameraStream(0)
camera2 = CameraStream(1)

# ───────────────────────────────
# Frame Generator with FPS Logging
# ───────────────────────────────
def generate_frames(camera_stream, name='cam'):
    prev_time = time.time()
    while True:
        start_time = time.time()
        frame = camera_stream.get_frame()
        if frame is None:
            continue

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ret:
          continue
        frame = buffer.tobytes()

        # Measure FPS and latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # in ms
        fps = 1 / (end_time - prev_time) if (end_time - prev_time) != 0 else 0
        prev_time = end_time
        print(f"[{name}] FPS: {fps:.2f} | Latency: {latency:.2f} ms")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # Adjust frame rate as needed

# ────────────────
# Flask Routes
# ────────────────

# Main page showing Camera 1 feed with status panel and Camera 2 feed below
# @app.route('/')
# def index():
#     return """
#     <html>
#       <head>
#         <title>Driver Monitoring Live Stream</title>
#         <meta name="viewport" content="width=device-width, initial-scale=1.0" />
#         <style>
#           /* Make sure body stretches to fill the entire screen height */
#           html, body {
#             margin: 0;
#             padding: 0;
#           }
          
#           /* Apply the vertical gradient to the entire page */
#           body {
#             min-height: 100vh; /* ensures gradient extends fully */
#             font-family: Arial, sans-serif;
#             background: linear-gradient(to bottom, #7D33A3, #6882C5);
#             background-repeat: no-repeat;
#             /* You can optionally pin the gradient so it doesn't move when scrolling:
#             background-attachment: fixed; */
#           }

#           /* Container to center the two cards side-by-side */
#           .container {
#             display: flex;
#             justify-content: center;
#             align-items: flex-start;
#             gap: 40px;         /* Space between the two cards */
#             padding: 40px;     /* Space from the edges of the page */
#           }

#           /* The “card-like” styling for each camera feed/status panel */
#           .card {
#             background-color: rgba(255, 255, 255, 0.8);  /* White w/ transparency */
#             border-radius: 10px;                         /* Rounded corners */
#             box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);      /* Subtle drop-shadow */
#             padding: 20px;
#             width: 700px;  /* Adjust width as needed */
#           }
#           .card h2 {
#             margin-top: 0;
#           }

#           /* Positioning for the camera feed and overlay images */
#           .camera-container {
#             position: relative;
#             display: inline-block;
#           }
#           .camera-container img {
#             display: block;
#             width: 640px;
#             height: auto;
#           }
#           .overlay {
#             position: absolute;
#             top: 0;
#             left: 0;
#             pointer-events: none;
#             width: 640px;
#             height: auto;
#           }
#         </style>
#         <script>
#           // Poll /status every 1 second
#           function updateStatus() {
#             fetch('/status')
#               .then(response => response.json())
#               .then(data => {
#                 // Update Camera 1 status panel using data.camera1
#                 document.getElementById("statusInfo").innerHTML = `
#                   <div style="font-family: Arial, sans-serif;">
#                     <strong style="font-size: 18px;">Activity and hands detection:</strong>
#                     <hr>
#                     <h3 style="margin: 0;">Per Frame Prediction</h3>
#                     <p style="margin: 0;">
#                       <b>Driver Activity:</b> ${data.camera1.per_frame_driver_activity}<br>
#                       <b>Hands-on-Wheel:</b> ${data.camera1.per_frame_hands_on_wheel}
#                     </p>
#                     <hr>
#                     <h3 style="margin: 0;">State Monitoring</h3>
#                     <p style="margin: 0;">
#                       <b>Majority Driver State:</b> ${data.camera1.majority_driver_state}<br>
#                       <b>System Alert:</b> ${data.camera1.system_alert}<br>
#                       <b>Hands Monitoring:</b> ${data.camera1.hands_monitoring}<br>
#                       <b>Hands Monitoring Confidence:</b> ${data.camera1.hands_monitoring_confidence}
#                     </p>
#                   </div>
#                 `;  
#                 // Update Camera 2 status panel using data.camera2
#                 document.getElementById("statusInfo2").innerHTML = `
#                   <div style="font-family: Arial, sans-serif;">
#                     <strong style="font-size: 18px;">Fatigue detection</strong>
#                     <hr>
#                     <h3 style="margin: 0;">Gaze Detection</h3>
#                     <p style="margin: 0;">
#                       <b>Position:</b> ${data.camera2.gaze_center}<br>
#                       <b>Status:</b> ${data.camera2.gaze_status}
#                     </p>
#                     <hr>
#                     <h3 style="margin: 0;">Head Movement</h3>
#                     <p style="margin: 0;">
#                       <b>Pitch:</b> ${data.camera2.pitch}<br>
#                       <b>Yaw:</b> ${data.camera2.yaw}<br>
#                       <b>Roll:</b> ${data.camera2.roll}<br>
#                       <b>Head Status:</b> ${data.camera2.head_status}
#                     </p>
#                     <hr>
#                     <h3 style="margin: 0;">Distraction</h3>
#                     <p style="margin: 0;">
#                       ${data.camera2.distraction}
#                     </p>
#                     <hr>
#                     <h3 style="margin: 0;">Drowsiness Detection</h3>
#                     <p style="margin: 0;">
#                       <b>Blinks:</b> ${data.camera2.blinks}<br>
#                       <b>Microsleep Duration:</b> ${data.camera2.microsleep_duration}<br>
#                       <b>Yawns:</b> ${data.camera2.yawns}<br>
#                       <b>Yawn Duration:</b> ${data.camera2.yawn_duration}<br>
#                       <b>Blinks Per Minute:</b> ${data.camera2.blinks_per_minute}<br>
#                       <b>Yawns Per Minute:</b> ${data.camera2.yawns_per_minute}<br>
#                       <b>Alert:</b> ${data.camera2.alert}
#                     </p>
#                   </div>
#                 `; 
#               })
#               .catch(err => {
#                 console.error("Error fetching status:", err);
#               });
#           }
#           // Update every 1 second (1000 ms)
#           setInterval(updateStatus, 1000);
#           window.onload = updateStatus;
#         </script>
#       </head>

#       <body>
#         <div style="display: flex; align-items: flex-start; justify-content: center; gap: 40px; padding: 40px;">
         
#           <!-- Left column: Camera 1 on top, status panel beneath -->
#           <div style="background-color: rgba(255, 255, 255, 0.8); border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.2); padding: 20px; width: 700px;">
#             <h2 style="margin: 0;">Camera 1 Feed</h2>
#             <img src="/video_feed1" style="border: 1px solid black; max-width: 640px; display: block; margin-top: 10px;"/>
#             <div id="statusInfo" style="margin-top: 20px; border: 1px solid #ccc; padding: 10px;">
#               <!-- status panel updates here -->
#             </div>
#           </div>
#           <!-- Right column: Camera 2 on top, status panel beneath -->
#           <div style="background-color: rgba(255, 255, 255, 0.8); border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.2); padding: 20px; width: 700px;">
#             <h2 style="margin: 0;">Camera 2 Feed</h2>
#             <img src="/video_feed2" style="border: 1px solid black; max-width: 640px; display: block; margin-top: 10px;"/>
#             <div id="statusInfo2" style="margin-top: 20px; border: 1px solid #ccc; padding: 10px;">
#               <!-- Status panel for Camera 2 updates here -->
#             </div>
#           </div>
#         </div>
#       </body>
#     </html>
#     """

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Driver Monitoring Live Stream</title>
        <!-- Responsive meta tag so the layout scales to device width -->
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        
        <style>
          /* Reset and ensure 100% height for html/body */
          html, body {
            margin: 0;
            padding: 0;
            
          }

          /* Make the gradient cover the entire visible area */
          body {
            min-height: 100vh;
            font-family: Arial, sans-serif;
            background: linear-gradient(to bottom, #7D33A3, #6882C5);
            background-repeat: no-repeat;
            background-size: cover;  /* Key: fill entire area */
            /* background-attachment: fixed;  <-- optional (often ignored on mobile) */
          }

          /* Flex container that can wrap to avoid horizontal scrolling */
          .container {
            display: flex;
            flex-wrap: wrap;        /* Allows cards to stack on small screens */
            justify-content: center;
            align-items: flex-start;
            gap: 40px;
            padding: 20px;
          }

          /* Card styling with a max-width for responsiveness */
          .card {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            padding: 20px;
            width: 100%;           /* Take full width on narrow screens */
            max-width: 700px;      /* But don’t exceed 700px on larger screens */
            box-sizing: border-box; /* Ensure padding doesn’t overflow */
          }
          .card h2 {
            margin-top: 0;
          }

          /* Camera container and images: fully responsive */
          .camera-container {
            position: relative;
            width: 100%;
            max-width: 640px;   /* Same ratio as your feed’s resolution */
            margin: 0 auto;     /* Center images within the card */
          }
          .camera-container img {
            display: block;
            width: 100%;        /* Scale down on small screens */
            height: auto;
          }
          .overlay {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            width: 100%;        /* Match the underlying image width */
            height: auto;
          }

          /* Simple styling for status panels */
          .status-panel {
            margin-top: 20px;
            border: 1px solid #ccc;
            padding: 10px;
          }
        </style>
        
        <script>
          // Poll /status every 1 second
          function updateStatus() {
            fetch('/status')
              .then(response => response.json())
              .then(data => {
                // Update Camera 1 status panel
                document.getElementById("statusInfo").innerHTML = `
                  <div>
                    <strong style="font-size: 18px;">Activity and hands detection:</strong>
                    <hr>
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
          setInterval(updateStatus, 1000);
          window.onload = updateStatus;
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



# Endpoint for camera 1 stream
@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(camera1, 'cam1'), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint for camera 2 stream
@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(camera2, 'cam2'), mimetype='multipart/x-mixed-replace; boundary=frame')

#status endpoint to read JSON files and return their contents
@app.route('/status')
def status():
    file1 = "status.json"
    file2 = "status_driver_fatigue.json"
    
    # Read Camera 1 data from file1
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
    
    # Read Camera 2 data from file2
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
    
    combined = {
        "camera1": data1,
        "camera2": data2
    }
    return jsonify(combined)



# ───────────────────────────────
# Start Waitress Production Server
# ───────────────────────────────
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000, threads=8)
