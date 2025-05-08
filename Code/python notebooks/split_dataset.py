import os
import shutil
import random
from pathlib import Path

# Config
RAW_DIR = "../data/raw"
OUTPUT_DIR = "../data"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
VAL_SPLIT = 0.2
SEED = 42

# Supported class names (normalized for folder names)
CLASS_MAP = {
    "Heracleum_sosnowskyi_Manden": "Heracleum_sosnowskyi",
    "Heracleum_mantegazzianum_Sommier": "Heracleum_mantegazzianum",
    "Heracleum_persicum_Desf": "Heracleum_persicum"
}

random.seed(SEED)

# Step 1: Create class directories
for split in ["train", "val"]:
    for cls in CLASS_MAP.values():
        split_dir = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

# Step 2: Classify and shuffle images
classified_images = {cls: [] for cls in CLASS_MAP.values()}

for fname in os.listdir(RAW_DIR):
    for key, label in CLASS_MAP.items():
        if key in fname:
            classified_images[label].append(fname)
            break

# Step 3: Split and copy
for label, files in classified_images.items():
    random.shuffle(files)
    val_count = int(len(files) * VAL_SPLIT)
    val_files = files[:val_count]
    train_files = files[val_count:]

    for f in train_files:
        src = os.path.join(RAW_DIR, f)
        dst = os.path.join(TRAIN_DIR, label, f)
        shutil.copy2(src, dst)

    for f in val_files:
        src = os.path.join(RAW_DIR, f)
        dst = os.path.join(VAL_DIR, label, f)
        shutil.copy2(src, dst)

print("Dataset successfully split into train/ and val/ folders.")
