from pathlib import Path
import shutil

DATASET_DIR = Path(r"Solar-Final2-yolov8")

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

for split in SPLITS:

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    if not image_dir.exists():
        print(f"[WARNING] Missing: {image_dir}")
        continue

    no_annotation = []

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        # No label file
        if not label_path.exists():
            no_annotation.append(image_path)
            continue

        # Empty label file
        if label_path.stat().st_size == 0:
            no_annotation.append(image_path)

    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    print(f"Images without annotations: {len(no_annotation)}")

    for image in no_annotation[:20]:
        print(" ", image.name)

    # Save list
    output_file = DATASET_DIR / f"{split}_unannotated.txt"

    with open(output_file, "w") as f:
        for image in no_annotation:
            f.write(str(image) + "\n")

    print(f"List saved to: {output_file}")