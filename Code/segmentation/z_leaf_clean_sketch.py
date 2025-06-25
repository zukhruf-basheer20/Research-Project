import cv2
import numpy as np
import os
from pathlib import Path

# === CONFIG ===
INPUT_DIR = Path("../data/Leaf_Final_Segmented")  # input = clean popped leaf PNGs with alpha
OUTPUT_DIR = Path("../data/Leaf_Clean_Sketch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"🔍 Found {len(image_files)} images in: {INPUT_DIR}")

for i, fname in enumerate(image_files, 1):
    try:
        img_path = INPUT_DIR / fname
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            print(f"⚠️ No alpha in {fname}, skipping.")
            continue

        # === Step 1: apply mask to image ===
        leaf_only = cv2.bitwise_and(bgr, bgr, mask=alpha)

        # === Step 2: convert to gray ===
        gray = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2GRAY)

        # === Step 3: blur to denoise ===
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # === Step 4: Canny edge on the masked leaf only ===
        edges = cv2.Canny(blur, 50, 150)

        # === Step 5: optional edge thickening ===
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # === Step 6: apply mask so edges are ONLY inside leaf ===
        edges_masked = cv2.bitwise_and(edges, edges, mask=alpha)

        # === Step 7: invert to get sketch style: white lines on black ===
        sketch = cv2.bitwise_not(edges_masked)

        out_fname = OUTPUT_DIR / f"{Path(fname).stem}_clean_sketch.png"
        cv2.imwrite(str(out_fname), sketch)

        print(f"✅ {i}/{len(image_files)}: {fname} -> {out_fname.name}")

    except Exception as e:
        print(f"❌ Error with {fname}: {e}")

print(f"\n🌟 DONE: Clean leaf sketches in {OUTPUT_DIR}")
