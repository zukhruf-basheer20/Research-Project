import cv2
import numpy as np
import os
from pathlib import Path

# === CONFIG ===
INPUT_DIR = Path("../data/Leaf_HSV_Clean")  # output of step 1
OUTPUT_DIR = Path("../data/Leaf_Final_Segmented")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === PROCESS ===
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"🔍 Found {len(image_files)} images in: {INPUT_DIR}")

for i, fname in enumerate(image_files, 1):
    try:
        img_path = INPUT_DIR / fname
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        # Use alpha if available to find green region
        if img.shape[2] == 4:
            green_mask = img[:, :, 3]
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array([20,40,40]), np.array([85,255,255]))

        # Edge detection on the cleaned image
        edges = cv2.Canny(img[:, :, :3], 100, 200)
        edges_masked = cv2.bitwise_and(edges, edges, mask=green_mask)
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges_masked, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)

            leaf_mask = np.zeros_like(green_mask)
            cv2.drawContours(leaf_mask, [hull], -1, 255, -1)
            leaf_mask_filled = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            leaf_only = np.zeros_like(img[:, :, :3])
            leaf_only[leaf_mask_filled == 255] = img[:, :, :3][leaf_mask_filled == 255]

            b, g, r = cv2.split(leaf_only)
            rgba = cv2.merge([b, g, r, leaf_mask_filled])

            out_fname = OUTPUT_DIR / f"{Path(fname).stem}_leafpop.png"
            cv2.imwrite(str(out_fname), rgba)

            print(f"✅ {i}/{len(image_files)}: {fname} -> {out_fname.name}")

        else:
            print(f"⚠️  No clear leaf found in {fname}")

    except Exception as e:
        print(f"❌ Error with {fname}: {e}")

print(f"\n🌟 Done: final segmented leaves in {OUTPUT_DIR}")
