
from ultralytics import YOLO
from picamera2 import Picamera2
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import time
import threading

# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = Path(
    "/home/oculus/Solaris/runs_backup/runs/yolo26n_custom/weights/best.pt"
)

CONFIDENCE = 0.40

IMG_SIZE = 416

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_FPS = 30

JPEG_QUALITY = 75

# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Solaris AI",
    description="YOLO26n Real-Time Solar Defect Detection",
    version="1.0.0"
)

# ==================================================
# GLOBAL VARIABLES
# ==================================================

model = None

picam2 = None

latest_frame = None

frame_lock = threading.Lock()

stop_event = threading.Event()

detection_thread = None

camera_status = False

fps_value = 0.0

detection_count = 0

# ==================================================
# LOAD YOLO26n MODEL
# ==================================================

def load_model():

    global model

    print("[INFO] Loading custom YOLO26n model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    print("[INFO] Model loaded successfully.")

    print("[INFO] Model path:", MODEL_PATH)

    print("[INFO] Classes:", model.names)


# ==================================================
# INITIALIZE RASPBERRY PI CAMERA 2
# ==================================================

def initialize_camera():

    global picam2
    global camera_status

    print("[INFO] Initializing Raspberry Pi Camera 2...")

    picam2 = Picamera2()

    camera_config = picam2.create_video_configuration(
        main={
            "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
            "format": "RGB888"
        },
        controls={
            "FrameRate": CAMERA_FPS
        },
        buffer_count=4
    )

    picam2.configure(camera_config)

    picam2.start()

    # Allow camera exposure and sensor to stabilize
    time.sleep(2)

    camera_status = True

    print("[INFO] Raspberry Pi Camera 2 started.")


# ==================================================
# YOLO26n DETECTION LOOP
# ==================================================

def detection_loop():

    global latest_frame
    global fps_value
    global detection_count

    print("[INFO] Starting YOLO26n detection...")

    prev_time = time.time()

    while not stop_event.is_set():

        # ------------------------------------------
        # CAPTURE FRAME USING PICAMERA2
        # ------------------------------------------

        try:

            rgb_frame = picam2.capture_array("main")

        except Exception as error:

            print("[ERROR] Camera capture failed:", error)

            time.sleep(0.1)

            continue

        # ------------------------------------------
        # RGB TO BGR CONVERSION
        # ------------------------------------------

        frame = cv2.cvtColor(
            rgb_frame,
            cv2.COLOR_RGB2BGR
        )

        # ------------------------------------------
        # YOLO26n INFERENCE
        # ------------------------------------------

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )

        result = results[0]

        # ------------------------------------------
        # DRAW BOUNDING BOXES
        # ------------------------------------------

        annotated = result.plot()

        # ------------------------------------------
        # FPS CALCULATION
        # ------------------------------------------

        current_time = time.time()

        elapsed = current_time - prev_time

        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        prev_time = current_time

        fps_value = fps

        # ------------------------------------------
        # COUNT DETECTIONS
        # ------------------------------------------

        detection_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        # ------------------------------------------
        # DISPLAY FPS
        # ------------------------------------------

        cv2.putText(
            annotated,
            f"YOLO26n | FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ------------------------------------------
        # DISPLAY DETECTION COUNT
        # ------------------------------------------

        cv2.putText(
            annotated,
            f"Detections: {detection_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ------------------------------------------
        # UPDATE SHARED FRAME
        # ------------------------------------------

        with frame_lock:

            latest_frame = annotated.copy()

    print("[INFO] Detection loop stopped.")


# ==================================================
# VIDEO STREAM GENERATOR
# ==================================================

def generate_frames():

    while not stop_event.is_set():

        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.05)

            continue

        # ------------------------------------------
        # JPEG ENCODING
        # ------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )

        if not success:
            continue

        # ------------------------------------------
        # MJPEG STREAM
        # ------------------------------------------

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ==================================================
# STARTUP EVENT
# ==================================================

@app.on_event("startup")
def startup_event():

    global detection_thread

    print("\n====================================")
    print("          SOLARIS AI")
    print("     YOLO26n DETECTION SERVER")
    print("====================================\n")

    stop_event.clear()

    load_model()

    initialize_camera()

    detection_thread = threading.Thread(
        target=detection_loop,
        daemon=True
    )

    detection_thread.start()

    print("[INFO] Detection server initialized.")


# ==================================================
# SHUTDOWN EVENT
# ==================================================

@app.on_event("shutdown")
def shutdown_event():

    global picam2
    global camera_status

    print("[INFO] Shutting down Solaris AI...")

    stop_event.set()

    if detection_thread is not None:

        detection_thread.join(timeout=5)

    if picam2 is not None:

        picam2.stop()

        picam2.close()

        picam2 = None

    camera_status = False

    print("[INFO] Camera released.")

    print("[INFO] Server stopped.")


# ==================================================
# DASHBOARD
# ==================================================

@app.get("/", response_class=HTMLResponse)
def dashboard():

    return """
    <!DOCTYPE html>

    <html lang="en">

    <head>

        <meta charset="UTF-8">

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <title>SOLARIS AI | YOLO26n</title>

        <style>

            * {
                box-sizing: border-box;
            }

            body {

                margin: 0;

                padding: 25px;

                background: #10141c;

                color: white;

                font-family: Arial, sans-serif;

            }

            .header {

                text-align: center;

                margin-bottom: 30px;

            }

            .header h1 {

                color: #00e676;

                font-size: 32px;

                letter-spacing: 3px;

                margin-bottom: 8px;

            }

            .header p {

                color: #9ca3af;

                font-size: 14px;

            }

            .dashboard {

                max-width: 1100px;

                margin: auto;

            }

            .status-container {

                display: flex;

                justify-content: space-between;

                align-items: center;

                background: #1c2430;

                padding: 15px 20px;

                border-radius: 10px;

                margin-bottom: 20px;

                flex-wrap: wrap;

                gap: 10px;

            }

            .status {

                color: #00e676;

                font-weight: bold;

            }

            .model {

                color: #aab4c3;

                font-size: 13px;

            }

            .video-container {

                background: #1c2430;

                padding: 12px;

                border-radius: 12px;

                border: 1px solid #303b4b;

            }

            .video-container img {

                display: block;

                width: 100%;

                height: auto;

                border-radius: 8px;

            }

            .footer {

                text-align: center;

                margin-top: 25px;

                color: #737f90;

                font-size: 12px;

            }

            @media(max-width:600px) {

                body {
                    padding: 12px;
                }

                .header h1 {
                    font-size: 25px;
                }

            }

        </style>

    </head>

    <body>

        <div class="dashboard">

            <div class="header">

                <h1>SOLARIS AI</h1>

                <p>
                    YOLO26n Real-Time Solar Defect Detection
                </p>

            </div>

            <div class="status-container">

                <div class="status">
                    ● LIVE DETECTION
                </div>

                <div class="model">
                    MODEL: CUSTOM YOLO26n | CAMERA: PI CAMERA 2
                </div>

            </div>

            <div class="video-container">

                <img
                    src="/video_feed"
                    alt="Live Detection Stream"
                >

            </div>

            <div class="footer">

                SOLARIS AI | EDGE AI VISION SYSTEM

            </div>

        </div>

    </body>

    </html>
    """


# ==================================================
# LIVE VIDEO ENDPOINT
# ==================================================

@app.get("/video_feed")
def video_feed():

    return StreamingResponse(

        generate_frames(),

        media_type=(
            "multipart/x-mixed-replace; boundary=frame"
        )

    )


# ==================================================
# HEALTH CHECK API
# ==================================================

@app.get("/health")
def health():

    return {

        "status": "running",

        "model": str(MODEL_PATH),

        "device": "CPU",

        "camera": (
            "active"
            if camera_status
            else "inactive"
        ),

        "fps": round(fps_value, 2),

        "detections": detection_count

    }


# ==================================================
# RUN SERVER
# ==================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1
    )
