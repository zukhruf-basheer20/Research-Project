#This script check from broken images in data set 
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import shutil

# List of folders to check
root_dirs = [
    Path("../../data/train/Heracleum_mantegazzianum"),
    Path("../../data/train/Heracleum_sosnowskyi"),
    Path("../../data/train/Heracleum_persicum"),
    Path("../../data/val/Heracleum_mantegazzianum"),
    Path("../../data/val/Heracleum_sosnowskyi"),
    Path("../../data/val/Heracleum_persicum"),
]


# Create broken dir
BROKEN_DIR = Path("../../data/broken")
BROKEN_DIR.mkdir(parents=True, exist_ok=True)

# Check each image: verify and open it
for folder in root_dirs:
    print(f"🔍 Checking {folder}")
    if not folder.exists():
        print(f"⚠️ Folder not found: {folder}")
        continue

    for img_path in folder.glob("*"):
        try:
            with Image.open(img_path) as img:
                img.load()  # fully decode image, not just verify header
        except (UnidentifiedImageError, OSError) as e:
            print(f"❌ Corrupt image: {img_path.name}")
            shutil.move(str(img_path), BROKEN_DIR / img_path.name)
