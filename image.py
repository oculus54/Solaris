from ultralytics import YOLO
from pathlib import Path
import cv2

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "weg" / "spectre.pt"
# Change the above if your .pt file is elsewhere.

IMAGE_PATH = BASE_DIR / "best.png"

OUTPUT_DIR = BASE_DIR / "detections"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIDENCE = 0.40
IMG_SIZE = 640

# --------------------------------------------------
# Load model
# --------------------------------------------------

print("[INFO] Loading model...")

model = YOLO(str(MODEL_PATH))

print("[INFO] Model loaded.")
print("[INFO] Model:", MODEL_PATH)
print("[INFO] Image:", IMAGE_PATH)

# --------------------------------------------------
# Run inference
# --------------------------------------------------

results = model.predict(
    source=str(IMAGE_PATH),
    imgsz=IMG_SIZE,
    conf=CONFIDENCE,
    device=0,
    verbose=False
)

# --------------------------------------------------
# Process result
# --------------------------------------------------

result = results[0]

# YOLO automatically draws:
# - bounding boxes
# - class names
# - confidence scores

annotated = result.plot()

# --------------------------------------------------
# Save output
# --------------------------------------------------

output_path = OUTPUT_DIR / "result.jpg"

cv2.imwrite(
    str(output_path),
    annotated
)

print("[INFO] Detection complete.")
print("[INFO] Output:", output_path)

# --------------------------------------------------
# Print detections
# --------------------------------------------------

if result.boxes is not None:

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = model.names[class_id]

        print(
            f"Detected: {class_name} "
            f"| Confidence: {confidence:.3f} "
            f"| Box: ({x1:.1f}, {y1:.1f}, "
            f"{x2:.1f}, {y2:.1f})"
        )

else:

    print("[INFO] No detections.")
