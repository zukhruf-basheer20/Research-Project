import os
import cv2
import subprocess
import numpy as np
from pathlib import Path

# === CONFIG ===
ROOT_DIR = Path(__file__).resolve().parent
GSAM_DIR = ROOT_DIR.parent / "segmentation" / "Grounded-Segment-Anything"
IMAGE_DIR = ROOT_DIR.parent / "data" / "Filtered_EffiecientnNetB0_V6"
TEMP_DIR = ROOT_DIR.parent / "data" / "leaf_SAM_for_outline"
OUTPUT_DIR = ROOT_DIR.parent / "data" / "outlined_with_sam_applied_to_Filtered_EffiecientnNetB0_V6"

GSAM_SCRIPT = GSAM_DIR / "grounded_sam_demo.py"
CONFIG = GSAM_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
SAM_CKPT = GSAM_DIR / "weights" / "sam_vit_h_4b8939.pth"
DINO_CKPT = GSAM_DIR / "weights" / "groundingdino_swint_ogc.pth"

TEMP_MASK = TEMP_DIR / "mask.jpg"
TEMP_RAW = TEMP_DIR / "raw_image.jpg"

# Ensure output directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === MAIN LOOP ===
for img_path in IMAGE_DIR.glob("*.jpg"):
    base_name = img_path.stem
    print(f"\n🔄 Processing: {base_name}")

    # STEP 1 — Run Grounded SAM
    try:
        subprocess.run(
            [
                "python", str(GSAM_SCRIPT),
                "--config", str(CONFIG),
                "--grounded_checkpoint", str(DINO_CKPT),
                "--sam_checkpoint", str(SAM_CKPT),
                "--input_image", str(img_path),
                "--output_dir", str(TEMP_DIR),
                "--text_prompt", "leaf",
                "--box_threshold", "0.3",
                "--text_threshold", "0.25"
            ],
            cwd=GSAM_DIR,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"SAM failed for {base_name}")
        print(e.stderr)
        continue

    # STEP 2 — Check mask and raw image exist
    if not TEMP_MASK.exists() or not TEMP_RAW.exists():
        print(f"Missing output for {base_name}, skipping.")
        continue

    # # STEP 3 — Load images
    image = cv2.imread(str(TEMP_RAW), cv2.IMREAD_COLOR)
    # mask = cv2.imread(str(TEMP_MASK), cv2.IMREAD_GRAYSCALE)

    # if image is None or mask is None:
    #     print(f"Failed to load raw/mask for {base_name}")
    #     continue

    # # STEP 4 — Resize mask if needed
    # if mask.shape != image.shape[:2]:
    #     print(f"[INFO] Resizing mask for {base_name} to match image dimensions...")
    #     mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # # STEP 5 — Threshold mask to binary
    # _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((3, 3), np.uint8)
    # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # STEP 3 — Load mask in COLOR
    mask_color = cv2.imread(str(TEMP_MASK), cv2.IMREAD_COLOR)

    if mask_color is None:
        print(f"Failed to load mask for {base_name}")
        continue

    # STEP 4 — Resize mask if needed
    if mask_color.shape[:2] != image.shape[:2]:
        print(f"[INFO] Resizing mask for {base_name} to match image dimensions...")
        mask_color = cv2.resize(mask_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert to HSV to isolate purple background
    hsv = cv2.cvtColor(mask_color, cv2.COLOR_BGR2HSV)

    # Define purple background range (adjust if needed)
    lower_purple = np.array([120 - 10, 50, 50])
    upper_purple = np.array([150 + 10, 255, 255])

    # Mask for purple background
    bg_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Invert to get foreground regions
    binary_mask = cv2.bitwise_not(bg_mask)

    # STEP 5 — Clean with morphology
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)


    # STEP 6 — Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found for {base_name}, skipping.")
        continue

    # STEP 7 — Create a blank black canvas for the outline
    outline_canvas = np.ones_like(image)

    # STEP 8 — Draw contours on the blank canvas in white
    cv2.drawContours(outline_canvas, contours, -1, (255, 255, 255), thickness=1)

    # STEP 9 — Save the outlined image
    output_path = OUTPUT_DIR / f"{base_name}_outline.png"
    cv2.imwrite(str(output_path), outline_canvas)

    print(f"✅ Saved outlined leaf: {output_path.name}")

print("✅ Saved outlined for leaf images using SAM mask")
