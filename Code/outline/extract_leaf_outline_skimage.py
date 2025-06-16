from pathlib import Path
import numpy as np
from skimage import io, measure, morphology, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.draw import polygon_perimeter
from skimage.morphology import closing, disk
import os

# === PATHS ===
ROOT_DIR = Path(__file__).resolve().parents[1]
MASK_DIR = ROOT_DIR / "data" / "Leaf_Segmented" / "masks"
OUTLINE_DIR = ROOT_DIR / "data" / "Leaf_Segmented" / "outlines1"

OUTLINE_DIR.mkdir(parents=True, exist_ok=True)

# === FIND MASKS ===
mask_files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith('.png')]
print(f"🔍 Found {len(mask_files)} mask images in: {MASK_DIR}")

for i, fname in enumerate(mask_files, 1):
    try:
        # Load mask as grayscale
        mask = io.imread(MASK_DIR / fname, as_gray=True)
        mask = mask > threshold_otsu(mask)  # binarize

        # 1️⃣ Remove tiny noise
        cleaned = morphology.remove_small_objects(mask, min_size=500)

        # 2️⃣ Fill holes inside leaf parts
        filled = morphology.remove_small_holes(cleaned, area_threshold=2000)

        # 3️⃣ Morphologically close to merge bits
        closed = closing(filled, disk(10))  # disk size = how much to merge, tweak!

        # Label connected regions
        labeled = measure.label(closed)
        props = measure.regionprops(labeled)

        if not props:
            print(f"⚠️ No regions after cleaning: {fname}")
            continue

        # Pick biggest connected region
        largest_region = max(props, key=lambda x: x.area)
        leaf_mask = (labeled == largest_region.label)

        # 4️⃣ Find contour
        contours = measure.find_contours(leaf_mask, 0.5)
        if not contours:
            print(f"⚠️ No contour for main region in {fname}")
            continue

        main_contour = max(contours, key=len)

        # 5️⃣ Create blank black image & draw
        outline = np.zeros_like(mask, dtype=np.uint8)
        rr, cc = polygon_perimeter(
            np.round(main_contour[:, 0]).astype(int),
            np.round(main_contour[:, 1]).astype(int),
            shape=outline.shape
        )
        outline[rr, cc] = 255

        # Save
        out_path = OUTLINE_DIR / f"{Path(fname).stem}_outline.png"
        io.imsave(out_path, img_as_ubyte(outline))

        print(f"✅ Robust outline saved for {fname} ({i}/{len(mask_files)})")

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

print(f"\n🟢 All solid outlines saved to: {OUTLINE_DIR}")
