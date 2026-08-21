from pathlib import Path
from collections import Counter

# Change this to your dataset path
DATASET_DIR = Path("SolarDataset")

# Class names from your dataset.yaml
CLASS_NAMES = [
    "bird-drop",
    "dust",
    "physical-damage",
    "electrical-damage",
    # Add the exact classes from your data.yaml
]

SPLITS = ["train", "valid", "test"]

for split in SPLITS:
    labels_dir = DATASET_DIR / split / "labels"
    images_dir = DATASET_DIR / split / "images"

    if not labels_dir.exists():
        print(f"\n[WARNING] {split}/labels not found")
        continue

    # Count images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    # Count class occurrences
    class_counter = Counter()

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                class_id = int(line.split()[0])

                if class_id < len(CLASS_NAMES):
                    class_counter[CLASS_NAMES[class_id]] += 1
                else:
                    class_counter[f"class_{class_id}"] += 1

    print("\n" + "=" * 50)
    print(f"{split.upper()}")
    print("=" * 50)

    print(f"Total images: {len(image_files)}")
    print(f"Total annotations: {sum(class_counter.values())}")

    for class_name in CLASS_NAMES:
        print(f"{class_name:25s}: {class_counter[class_name]}")

print("\n" + "=" * 50)
print("COMBINED TRAIN + VALID + TEST")
print("=" * 50)

total_counter = Counter()
total_images = 0

for split in SPLITS:
    labels_dir = DATASET_DIR / split / "labels"
    images_dir = DATASET_DIR / split / "images"

    if not labels_dir.exists():
        continue

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    total_images += sum(
        1 for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    )

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                class_id = int(line.split()[0])

                if class_id < len(CLASS_NAMES):
                    total_counter[CLASS_NAMES[class_id]] += 1
                else:
                    total_counter[f"class_{class_id}"] += 1

print(f"Total images: {total_images}")
print(f"Total annotations: {sum(total_counter.values())}")

for class_name in CLASS_NAMES:
    print(f"{class_name:25s}: {total_counter[class_name]}")
