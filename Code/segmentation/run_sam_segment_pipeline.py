import os
import cv2
import subprocess
import numpy as np
from pathlib import Path

# === CONFIG ===
ROOT_DIR = Path(__file__).resolve().parent
GSAM_DIR = ROOT_DIR / "Grounded-Segment-Anything"
IMAGE_DIR = ROOT_DIR.parent / "data" / "Filtered_EffiecientnNetB0_V6"
TEMP_DIR = ROOT_DIR.parent / "data" / "leaf_SAM"
OUTPUT_DIR = ROOT_DIR.parent / "data" / "segmented_with_sam_applied_to_Filtered_EffiecientnNetB0_V6"

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
        print(f"❌ SAM failed for {base_name}")
        print(e.stderr)
        continue

    # STEP 2 — Process mask + raw image
    if not TEMP_MASK.exists() or not TEMP_RAW.exists():
        print(f"⚠️ Missing output for {base_name}, skipping.")
        continue

    image = cv2.imread(str(TEMP_RAW), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(TEMP_MASK), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"❌ Failed to load raw/mask for {base_name}")
        continue

    # Resize mask to match image if needed
    if mask.shape != image.shape[:2]:
        print(f"[INFO] Resizing mask for {base_name} to match image dimensions...")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Threshold the mask to binary
    _, alpha = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    alpha = alpha.astype(np.uint8)

    # Merge channels with alpha
    b, g, r = cv2.split(image)
    rgba = cv2.merge([b, g, r, alpha])

    # Save final transparent PNG
    output_path = OUTPUT_DIR / f"{base_name}.png"
    cv2.imwrite(str(output_path), rgba)
    print(f"✅ Saved segmented leaf: {output_path.name}")
