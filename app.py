from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import threading
import time

app = Flask(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "./weg/spectre.pt"
CAMERA_INDEX = 0
IMG_SIZE = 640
CONFIDENCE = 0.40

# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

print("[INFO] Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("[INFO] Model loaded.")

# --------------------------------------------------
# Camera
# --------------------------------------------------

camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    raise RuntimeError("Could not open camera.")

# Lock prevents simultaneous camera access
camera_lock = threading.Lock()


def generate_frames():

    while True:

        with camera_lock:
            success, frame = camera.read()

        

        # --------------------------------------------------
        # YOLO inference
        # --------------------------------------------------

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            device=0,
            verbose=False
        )

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # --------------------------------------------------
        # Encode frame as JPEG
        # --------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        # MJPEG stream
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# --------------------------------------------------
# Web page
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# Video stream
# --------------------------------------------------

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":

    print("[INFO] Starting Flask server...")
    print("[INFO] Open http://127.0.0.1:5000 locally")
    print("[INFO] For LAN access use http://<YOUR-PC-IP>:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False
    )
