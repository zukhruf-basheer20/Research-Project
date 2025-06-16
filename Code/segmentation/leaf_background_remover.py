import cv2
import numpy as np
import os
from pathlib import Path

# === CONFIG ===
INPUT_DIR = Path("../data/leaf")  # <- your input folder
OUTPUT_DIR = Path("../data/Leaf_Segmented")
MASK_DIR = OUTPUT_DIR / "masks"
SEGMENTED_DIR = OUTPUT_DIR / "segmented"

# Create output dirs if needed
MASK_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)

# === PARAMETERS ===
# Adjust these for your dataset!
LOWER_GREEN = np.array([20, 40, 40])  # HSV lower bound for green
UPPER_GREEN = np.array([85, 255, 255])  # HSV upper bound for green

# === PROCESS EACH IMAGE ===
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"🔍 Found {len(image_files)} images in: {INPUT_DIR}")

for i, fname in enumerate(image_files, 1):
    try:
        img_path = INPUT_DIR / fname
        img = cv2.imread(str(img_path))

        # Convert to HSV (better for green detection)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create mask for green areas
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

        # Optional: Clean mask (remove noise)
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Create output: leaf with background removed (transparent PNG)
        result = cv2.bitwise_and(img, img, mask=mask_clean)

        # Optionally, make background white or transparent
        # Convert to 4-channel BGRA
        b, g, r = cv2.split(result)
        alpha = mask_clean
        rgba = cv2.merge([b, g, r, alpha])

        # === SAVE ===
        mask_path = MASK_DIR / f"{fname.split('.')[0]}_mask.png"
        result_path = SEGMENTED_DIR / f"{fname.split('.')[0]}_leaf.png"

        cv2.imwrite(str(mask_path), mask_clean)
        cv2.imwrite(str(result_path), rgba)

        print(f"✅ Processed {i}/{len(image_files)}: {fname}")

    except Exception as e:
        print(f"❌ Error processing {fname}: {str(e)}")

print(f"\n🟢 Done! Masks saved in: {MASK_DIR}")
print(f"🟢 Segmented leaves saved in: {SEGMENTED_DIR}")
