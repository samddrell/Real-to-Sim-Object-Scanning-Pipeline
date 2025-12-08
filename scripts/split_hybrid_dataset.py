#!/usr/bin/env python
"""
Split HYBRID bottle01 dataset (real + synthetic) into train/val for YOLO.

Assumes layout:

  HYBRID_ROOT/
    images/
      img001.jpg
      img002.jpg
      ...
    labels/
      img001.txt
      img002.txt
      ...

Creates:

  HYBRID_ROOT/images/train/
  HYBRID_ROOT/images/val/
  HYBRID_ROOT/labels/train/
  HYBRID_ROOT/labels/val/

and MOVES each image + its label into the appropriate split.

Adjust TRAIN_FRACTION below if you want a different split.

RUN:
python C:\Users\samdd\Documents\school\ML\Real-to-Sim-Object-Scanning-Pipeline\scripts\split_hybrid_dataset.py
"""

import random
from pathlib import Path
import shutil

# ====================== USER CONFIG ======================

# Root of the hybrid dataset on E:\
HYBRID_ROOT = Path(r"E:\datasets\bottle01\hybrid_datasets")

# Fractions for train/val split
TRAIN_FRACTION = 0.8   # 80% train
VAL_FRACTION = 0.2     # 20% val (must be 1 - TRAIN_FRACTION)
RANDOM_SEED = 42       # for reproducibility

# Valid image extensions to look for
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==================== END USER CONFIG ====================


def main():
    images_root = HYBRID_ROOT / "images"
    labels_root = HYBRID_ROOT / "labels"

    if not images_root.exists():
        raise SystemExit(f"Images folder not found: {images_root}")
    if not labels_root.exists():
        raise SystemExit(f"Labels folder not found: {labels_root}")

    # Destination folders
    images_train = images_root / "train"
    images_val = images_root / "val"
    labels_train = labels_root / "train"
    labels_val = labels_root / "val"

    # Create destination dirs if they don't exist
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all image files in images_root that are NOT in train/val already
    image_files = []
    for p in images_root.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(p)

    if not image_files:
        print(f"No top-level images found in {images_root}")
        print("Make sure your hybrid images are directly under 'images/', "
              "not already in 'images/train' or 'images/val'.")
        return

    print(f"Found {len(image_files)} images in {images_root}")

    # Filter out any images that don't have a matching label
    paired = []
    missing_labels = []
    for img_path in image_files:
        label_path = labels_root / (img_path.stem + ".txt")
        if label_path.exists():
            paired.append((img_path, label_path))
        else:
            missing_labels.append(img_path)

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images have no matching label .txt files:")
        for p in missing_labels:
            print(f"  - {p.name}")
        print("These will be skipped.")

    if not paired:
        print("No images with matching labels found. Aborting.")
        return

    print(f"{len(paired)} images have matching labels and will be split.")

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(paired)

    n_total = len(paired)
    n_train = int(round(n_total * TRAIN_FRACTION))
    n_val = n_total - n_train

    train_pairs = paired[:n_train]
    val_pairs = paired[n_train:]

    print(f"Train: {len(train_pairs)} images")
    print(f"Val:   {len(val_pairs)} images")

    # Move files
    def move_pair(img_src: Path, lbl_src: Path, img_dst_dir: Path, lbl_dst_dir: Path):
        img_dst = img_dst_dir / img_src.name
        lbl_dst = lbl_dst_dir / lbl_src.name

        print(f"  Moving {img_src.name} -> {img_dst_dir.name}/")
        shutil.move(str(img_src), str(img_dst))
        shutil.move(str(lbl_src), str(lbl_dst))

    print("\nMoving train files...")
    for img_src, lbl_src in train_pairs:
        move_pair(img_src, lbl_src, images_train, labels_train)

    print("\nMoving val files...")
    for img_src, lbl_src in val_pairs:
        move_pair(img_src, lbl_src, images_val, labels_val)

    print("\nDone!")
    print(f"Final structure should look like:")
    print(f"  {HYBRID_ROOT}/")
    print(f"    images/train/*.jpg")
    print(f"    images/val/*.jpg")
    print(f"    labels/train/*.txt")
    print(f"    labels/val/*.txt")


if __name__ == "__main__":
    main()
