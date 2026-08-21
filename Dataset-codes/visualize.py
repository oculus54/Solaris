from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(r"Solar-Final2-yolov8")

CLASS_NAMES = [
    "bird-drop",
    "dust",
    "physical-damage",
    "electrical-damage"
]

# Choose split
SPLIT = "train"

# Number of random images to inspect
NUM_IMAGES = 50

# Random seed for reproducibility
random.seed(42)


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = DATASET_DIR / SPLIT / "images"
LABEL_DIR = DATASET_DIR / SPLIT / "labels"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FIND IMAGES
# ============================================================

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

if not images:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

NUM_IMAGES = min(NUM_IMAGES, len(images))

selected_images = random.sample(images, NUM_IMAGES)


# ============================================================
# VISUALIZATION
# ============================================================

for index, image_path in enumerate(selected_images, start=1):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[WARNING] Could not read: {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Read annotations
    # --------------------------------------------------------

    annotations = []

    if label_path.exists():

        with open(label_path, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                # Convert normalized YOLO coordinates
                x_center *= image_width
                y_center *= image_height

                box_width *= image_width
                box_height *= image_height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)

                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                # Keep box inside image
                x1 = max(0, min(x1, image_width - 1))
                y1 = max(0, min(y1, image_height - 1))
                x2 = max(0, min(x2, image_width - 1))
                y2 = max(0, min(y2, image_height - 1))

                annotations.append(
                    (class_id, x1, y1, x2, y2)
                )

    # --------------------------------------------------------
    # Draw bounding boxes
    # --------------------------------------------------------

    for class_id, x1, y1, x2, y2 in annotations:

        if 0 <= class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"class_{class_id}"

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Label background
        text = class_name

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        text_y = max(y1, text_height + baseline)

        cv2.rectangle(
            image,
            (x1, text_y - text_height - baseline),
            (x1 + text_width, text_y),
            (255, 0, 0),
            -1
        )

        cv2.putText(
            image,
            text,
            (x1, text_y - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    plt.figure(figsize=(14, 9))

    plt.imshow(image)
    plt.axis("off")

    plt.title(
        f"{index}/{NUM_IMAGES} | "
        f"{SPLIT} | "
        f"{image_path.name} | "
        f"Boxes: {len(annotations)}"
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # User control
    # --------------------------------------------------------

    command = input(
        "\nPress ENTER for next image, "
        "'q' to quit: "
    ).strip().lower()

    if command == "q":
        break


print("\nVisualization finished.")