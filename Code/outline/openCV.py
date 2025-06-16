import cv2
import numpy as np
import os
from pathlib import Path

# === CONFIG ===
INPUT_DIR = Path("../data/leaf")  # your input folder with raw leaf images
OUTPUT_DIR = Path("../data/Leaf_Segmented1")
MASK_DIR = OUTPUT_DIR / "masks"
SEGMENTED_DIR = OUTPUT_DIR / "segmented"
OUTLINE_DIR = OUTPUT_DIR / "outlines"

# Create output dirs if needed
MASK_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
OUTLINE_DIR.mkdir(parents=True, exist_ok=True)

# === HSV THRESHOLDS ===
LOWER_GREEN = np.array([20, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

# === PROCESS ===
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"🔍 Found {len(image_files)} images in: {INPUT_DIR}")

for i, fname in enumerate(image_files, 1):
    try:
        img_path = INPUT_DIR / fname
        img = cv2.imread(str(img_path))

        # Convert to HSV & mask green
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # === SEGMENTED LEAF (TRANSPARENT) ===
        result = cv2.bitwise_and(img, img, mask=mask_clean)
        b, g, r = cv2.split(result)
        alpha = mask_clean
        rgba = cv2.merge([b, g, r, alpha])

        # === OUTLINE ===
        # Find all contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If multiple, keep the biggest one (your leaf)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            outline = np.zeros_like(mask_clean)
            cv2.drawContours(outline, [largest], -1, 255, 2)  # white outline, thickness=2
        else:
            outline = np.zeros_like(mask_clean)

        # === SAVE ===
        base = fname.split('.')[0]
        cv2.imwrite(str(MASK_DIR / f"{base}_mask.png"), mask_clean)
        cv2.imwrite(str(SEGMENTED_DIR / f"{base}_leaf.png"), rgba)
        cv2.imwrite(str(OUTLINE_DIR / f"{base}_outline.png"), outline)

        print(f"✅ Done {i}/{len(image_files)}: {fname}")

    except Exception as e:
        print(f"❌ Error with {fname}: {e}")

print(f"\n🟢 Masks: {MASK_DIR}")
print(f"🟢 Segmented: {SEGMENTED_DIR}")
print(f"🟢 Outlines: {OUTLINE_DIR}")
