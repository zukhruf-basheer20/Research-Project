import os
import shutil
import random
from pathlib import Path

def split_and_copy(source_dir, train_dir, val_dir, split_ratio=0.8):
    images = list(Path(source_dir).glob('*'))
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Ensure destination directories exist
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    # Copy train images
    for img in train_images:
        shutil.copy(img, Path(train_dir) / img.name)

    # Copy validation images
    for img in val_images:
        shutil.copy(img, Path(val_dir) / img.name)

# Define your source folders
leaf_source = "leaf"
no_leaf_source = "no_leaf"

# Destination folders
train_leaf = "train/leaf"
val_leaf = "val/leaf"
train_no_leaf = "train/no_leaf"
val_no_leaf = "val/no_leaf"

# Perform split
split_and_copy(leaf_source, train_leaf, val_leaf, split_ratio=0.8)
split_and_copy(no_leaf_source, train_no_leaf, val_no_leaf, split_ratio=0.8)

print("Folders organized! 🎉")
