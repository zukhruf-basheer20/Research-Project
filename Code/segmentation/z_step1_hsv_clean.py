import cv2
import numpy as np
import os
from pathlib import Path

# === CONFIG ===
INPUT_DIR = Path("../data/leaf")  # raw images
OUTPUT_DIR = Path("../data/Leaf_HSV_Clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === HSV THRESHOLDS (STRONGER GREEN FILTER) ===
LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([80, 255, 255])

image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"🔍 Found {len(image_files)} images in: {INPUT_DIR}")

for i, fname in enumerate(image_files, 1):
    try:
        img_path = INPUT_DIR / fname
        img = cv2.imread(str(img_path))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

        # Morphology to clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Remove small blobs (noise or background leaves)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
        min_area = 1000  # Only keep blobs bigger than this
        mask_filtered = np.zeros_like(mask_clean)

        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] >= min_area:
                mask_filtered[labels == j] = 255

        # Apply mask
        result = np.zeros_like(img)
        result[mask_filtered == 255] = img[mask_filtered == 255]

        # Add alpha channel
        b, g, r = cv2.split(result)
        rgba = cv2.merge([b, g, r, mask_filtered])

        out_fname = OUTPUT_DIR / f"{Path(fname).stem}_hsvclean.png"
        cv2.imwrite(str(out_fname), rgba)

        print(f"✅ {i}/{len(image_files)}: {fname} -> {out_fname.name}")

    except Exception as e:
        print(f"❌ Error with {fname}: {e}")

print(f"\n🌟 Done: HSV cleaned images in {OUTPUT_DIR}")
