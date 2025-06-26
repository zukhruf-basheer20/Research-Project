import os
import subprocess
from pathlib import Path

# ✅ Paths
ROOT_DIR = Path(__file__).resolve().parent
GSAM_DIR = ROOT_DIR / "Grounded-Segment-Anything"
IMAGE_DIR = ROOT_DIR.parent / "data" / "leaf"
OUTPUT_DIR = ROOT_DIR.parent / "data" / "leaf_SAM"

# ✅ Script and config files
GSAM_SCRIPT = GSAM_DIR / "grounded_sam_demo.py"
CONFIG = GSAM_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
SAM_CKPT = GSAM_DIR / "weights" / "sam_vit_h_4b8939.pth"
DINO_CKPT = GSAM_DIR / "weights" / "groundingdino_swint_ogc.pth"

# ✅ Create output folder if missing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Process each image
for img_path in IMAGE_DIR.glob("*.jpg"):
    print(f"🔄 Processing: {img_path.stem}")
    
    try:
        result = subprocess.run(
            [
                "python", str(GSAM_SCRIPT),
                "--config", str(CONFIG),
                "--grounded_checkpoint", str(DINO_CKPT),
                "--sam_checkpoint", str(SAM_CKPT),
                "--input_image", str(img_path),
                "--output_dir", str(OUTPUT_DIR),
                "--text_prompt", "leaf",
                "--box_threshold", "0.3",
                "--text_threshold", "0.25"
            ],
            cwd=GSAM_DIR,  # ← CRITICAL! Run inside Grounded-Segment-Anything folder
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Done: {img_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ SAM failed for {img_path.name}")
        print(e.stderr)
