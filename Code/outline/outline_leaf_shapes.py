from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import os

# === PATHS ===
ROOT_DIR = Path(__file__).resolve().parents[1]
MASK_DIR = ROOT_DIR / "data" / "Leaf_Segmented" / "masks"
OUTLINE_DIR = ROOT_DIR / "data" / "Leaf_Segmented" / "outlines"

OUTLINE_DIR.mkdir(parents=True, exist_ok=True)

# === FIND MASKS ===
mask_files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith('.png')]
print(f"🔍 Found {len(mask_files)} masks in: {MASK_DIR}")

for i, fname in enumerate(mask_files, 1):
    try:
        mask_path = MASK_DIR / fname
        mask = Image.open(mask_path).convert("L")

        # Find edges directly on binary mask
        edges = mask.filter(ImageFilter.FIND_EDGES)

        # Binarize: edges white, background black
        outline = edges.point(lambda p: 255 if p > 20 else 0)

        # Invert if needed — usually not necessary here
        outline = outline.convert("L")

        # Save
        out_path = OUTLINE_DIR / f"{Path(fname).stem}_outline.png"
        outline.save(out_path)

        print(f"✅ Outlined {i}/{len(mask_files)}: {fname}")

    except Exception as e:
        print(f"❌ Error processing {fname}: {str(e)}")

print(f"\n🟢 Clean outlines saved to: {OUTLINE_DIR}")
