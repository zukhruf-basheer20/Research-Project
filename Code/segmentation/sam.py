import os
import cv2
import torch
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# === CONFIG ===
INPUT_DIR = Path("../data/leaf")  # change this to your actual input
OUTPUT_DIR = Path("../data/Leaf_SAM_Segmented")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sam_checkpoint = "sam_vit_h_4b8939.pth"  # path to downloaded checkpoint
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD SAM MODEL ===
print("🔁 Loading SAM model...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
print("✅ SAM model loaded.")

# === PROCESS ALL IMAGES ===
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"🔍 Found {len(image_files)} images in: {INPUT_DIR}")

for i, fname in enumerate(image_files, 1):
    try:
        img_path = INPUT_DIR / fname
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"⚠️ Could not read {fname}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image_rgb)

        if not masks:
            print(f"⚠️ No masks found in {fname}")
            continue

        # Find largest mask
        biggest = max(masks, key=lambda x: x['area'])
        mask_array = biggest['segmentation'].astype(np.uint8) * 255

        # Apply mask
        masked_image = np.zeros_like(image_rgb)
        masked_image[mask_array == 255] = image_rgb[mask_array == 255]

        # Convert to RGBA
        r, g, b = cv2.split(masked_image)
        rgba = cv2.merge([b, g, r, mask_array])  # OpenCV uses BGR, so flip

        out_fname = OUTPUT_DIR / f"{Path(fname).stem}_samseg.png"
        cv2.imwrite(str(out_fname), rgba)

        print(f"✅ {i}/{len(image_files)}: {fname} -> {out_fname.name}")

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

print(f"\n🌿 All done! Segmented images saved in: {OUTPUT_DIR}")
