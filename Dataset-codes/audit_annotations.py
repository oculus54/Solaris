from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(r"Solar-Final2-yolov8")

SPLITS = ["train", "valid", "test"]

CLASS_NAMES = [
    "bird-drop",
    "dust",
    "physical-damage",
    "electrical-damage"
]

# YOLO normalized coordinate limits
MIN_BOX_SIZE = 0.001
MAX_BOX_SIZE = 1.0


# ============================================================
# AUDIT
# ============================================================

total_images = 0
total_annotations = 0

global_stats = Counter()

print("\n" + "=" * 70)
print("SOLAR PANEL DATASET ANNOTATION AUDIT")
print("=" * 70)


for split in SPLITS:

    labels_dir = DATASET_DIR / split / "labels"
    images_dir = DATASET_DIR / split / "images"

    if not labels_dir.exists():
        print(f"\n[WARNING] Missing: {labels_dir}")
        continue

    print(f"\n{'=' * 70}")
    print(f"{split.upper()}")
    print(f"{'=' * 70}")

    split_images = 0
    split_annotations = 0

    empty_labels = []
    malformed_labels = []
    invalid_class = []
    invalid_coordinates = []
    tiny_boxes = []
    oversized_boxes = []
    duplicate_boxes = []

    class_counts = Counter()

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_extensions = {
        ".jpg", ".jpeg", ".png",
        ".bmp", ".webp"
    }

    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    split_images = len(image_files)

    # --------------------------------------------------------
    # Check each label file
    # --------------------------------------------------------

    for label_file in labels_dir.glob("*.txt"):

        total_annotations_in_file = 0
        boxes = []

        try:
            with open(label_file, "r") as f:
                lines = f.readlines()

        except Exception as e:
            malformed_labels.append(
                (label_file.name, f"Cannot read file: {e}")
            )
            continue

        # Empty annotation file
        if len(lines) == 0:
            empty_labels.append(label_file.name)
            continue

        for line_number, line in enumerate(lines, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # YOLO format must contain 5 values
            if len(parts) != 5:
                malformed_labels.append(
                    (
                        label_file.name,
                        f"Line {line_number}: expected 5 values, got {len(parts)}"
                    )
                )
                continue

            try:
                class_id = int(parts[0])

                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:
                malformed_labels.append(
                    (
                        label_file.name,
                        f"Line {line_number}: non-numeric value"
                    )
                )
                continue

            total_annotations_in_file += 1

            # ------------------------------------------------
            # Class ID check
            # ------------------------------------------------

            if class_id < 0 or class_id >= len(CLASS_NAMES):

                invalid_class.append(
                    (
                        label_file.name,
                        line_number,
                        class_id
                    )
                )

            else:
                class_counts[class_id] += 1

            # ------------------------------------------------
            # Coordinate check
            # ------------------------------------------------

            values = [x, y, w, h]

            if any(v < 0 or v > 1 for v in values):

                invalid_coordinates.append(
                    (
                        label_file.name,
                        line_number,
                        values
                    )
                )

            # ------------------------------------------------
            # Tiny box check
            # ------------------------------------------------

            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:

                tiny_boxes.append(
                    (
                        label_file.name,
                        line_number,
                        w,
                        h
                    )
                )

            # ------------------------------------------------
            # Oversized box check
            # ------------------------------------------------

            if w > MAX_BOX_SIZE or h > MAX_BOX_SIZE:

                oversized_boxes.append(
                    (
                        label_file.name,
                        line_number,
                        w,
                        h
                    )
                )

            # Save box for duplicate checking
            boxes.append(
                (
                    class_id,
                    round(x, 6),
                    round(y, 6),
                    round(w, 6),
                    round(h, 6)
                )
            )

        split_annotations += total_annotations_in_file

        # ----------------------------------------------------
        # Duplicate boxes
        # ----------------------------------------------------

        if len(boxes) != len(set(boxes)):

            duplicates = len(boxes) - len(set(boxes))

            duplicate_boxes.append(
                (
                    label_file.name,
                    duplicates
                )
            )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(f"\nImages:              {split_images}")
    print(f"Annotations:         {split_annotations}")

    print("\nClass distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):

        print(
            f"  {class_id}: {class_name:20s}"
            f"{class_counts[class_id]:6d}"
        )

    print("\nPotential problems:")

    print(
        f"  Empty label files:       {len(empty_labels)}"
    )

    print(
        f"  Malformed annotations:   {len(malformed_labels)}"
    )

    print(
        f"  Invalid class IDs:       {len(invalid_class)}"
    )

    print(
        f"  Invalid coordinates:     {len(invalid_coordinates)}"
    )

    print(
        f"  Tiny bounding boxes:     {len(tiny_boxes)}"
    )

    print(
        f"  Oversized bounding boxes:{len(oversized_boxes)}"
    )

    print(
        f"  Duplicate boxes:         {len(duplicate_boxes)}"
    )

    # --------------------------------------------------------
    # Store globally
    # --------------------------------------------------------

    total_images += split_images
    total_annotations += split_annotations

    global_stats["empty"] += len(empty_labels)
    global_stats["malformed"] += len(malformed_labels)
    global_stats["invalid_class"] += len(invalid_class)
    global_stats["invalid_coordinates"] += len(invalid_coordinates)
    global_stats["tiny"] += len(tiny_boxes)
    global_stats["oversized"] += len(oversized_boxes)
    global_stats["duplicates"] += len(duplicate_boxes)

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    if malformed_labels:
        print("\nExample malformed annotations:")

        for item in malformed_labels[:10]:
            print(" ", item)

    if invalid_class:
        print("\nExample invalid class IDs:")

        for item in invalid_class[:10]:
            print(" ", item)

    if invalid_coordinates:
        print("\nExample invalid coordinates:")

        for item in invalid_coordinates[:10]:
            print(" ", item)

    if tiny_boxes:
        print("\nExample tiny boxes:")

        for item in tiny_boxes[:10]:
            print(" ", item)

    if duplicate_boxes:
        print("\nExample duplicate boxes:")

        for item in duplicate_boxes[:10]:
            print(" ", item)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET AUDIT SUMMARY")
print("=" * 70)

print(f"\nTotal images:       {total_images}")
print(f"Total annotations:  {total_annotations}")

print("\nPotential problems across dataset:")

for key, value in global_stats.items():
    print(f"  {key:20s}: {value}")

print("\nAudit complete.")