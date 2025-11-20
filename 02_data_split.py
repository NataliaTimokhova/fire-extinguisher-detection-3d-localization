# data_split.py
import os
import random
import shutil
from pathlib import Path

# CONFIG
DATASET_DIR = Path("FireExtinguisher.v2i.yolov11")
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"
SPLIT_RATIO = 1 / 3  # 1/3 of valid will become test

# Paths
valid_images = list((VALID_DIR / "images").glob("*.jpg")) + list((VALID_DIR / "images").glob("*.png"))
valid_labels = VALID_DIR / "labels"

# Create test folders if they don’t exist
(TEST_DIR / "images").mkdir(parents=True, exist_ok=True)
(TEST_DIR / "labels").mkdir(parents=True, exist_ok=True)

# Shuffle and split
random.seed(42)
random.shuffle(valid_images)
num_test = int(len(valid_images) * SPLIT_RATIO)
test_images = valid_images[:num_test]

print(f"Total valid images: {len(valid_images)}")
print(f"Moving {num_test} images to test...")

moved, missing_labels = 0, 0
for img_path in test_images:
    label_name = img_path.stem + ".txt"
    label_path = valid_labels / label_name

    # Move image
    shutil.move(str(img_path), str(TEST_DIR / "images" / img_path.name))

    # Move corresponding label
    if label_path.exists():
        shutil.move(str(label_path), str(TEST_DIR / "labels" / label_name))
    else:
        missing_labels += 1

    moved += 1

print("\nSplit complete.")
print(f"Valid images left: {len(os.listdir(VALID_DIR / 'images'))}")
print(f"Test images created: {len(os.listdir(TEST_DIR / 'images'))}")
if missing_labels:
    print(f"Warning: {missing_labels} labels were missing for some images.")
