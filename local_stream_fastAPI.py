import cv2
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Initialize webcam
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize frame (compression) - scale down to ~60%
            frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

            # Compress frame with JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
def index():
    html = """
    <html>
        <head>
            <title>Live Stream</title>
        </head>
        <body>
            <h1>Live Camera Feed</h1>
            <img src="/video_feed" width="720" />
        </body>
    </html>
    """
    return Response(content=html, media_type="text/html")


if __name__ == "__main__":
    uvicorn.run("local_stream_fastAPI:app", host="0.0.0.0", port=5000)
