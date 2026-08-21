from pathlib import Path
from collections import Counter

DATASET_DIR = Path("Solar-Final2-yolov8")

# Better: use the exact names/order from data.yaml
CLASS_NAMES = [
    "bird-drop",
    "dust",
    "physical-damage",
    "electrical-damage",
]

SPLITS = ["train", "valid", "test"]

for split in SPLITS:

    labels_dir = DATASET_DIR / split / "labels"

    class_image_count = Counter()
    total_images = 0

    for label_file in labels_dir.glob("*.txt"):

        total_images += 1
        classes_in_image = set()

        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                class_id = int(line.split()[0])
                classes_in_image.add(class_id)

        for class_id in classes_in_image:
            if class_id < len(CLASS_NAMES):
                class_image_count[CLASS_NAMES[class_id]] += 1

    print(f"\n{'='*45}")
    print(f"{split.upper()}")
    print(f"{'='*45}")

    print(f"Total labeled images: {total_images}")

    for class_name in CLASS_NAMES:
        count = class_image_count[class_name]
        percentage = (count / total_images * 100) if total_images else 0

        print(
            f"{class_name:25s}: "
            f"{count:5d} images ({percentage:.2f}%)"
        )