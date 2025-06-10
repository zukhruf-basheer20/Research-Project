import os
from pathlib import Path
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import UnidentifiedImageError

# === 🔧 Paths ===
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "Filtered_EffiecientnNetB0_V6"
MODEL_PATH = ROOT_DIR / "models" / "EffiecientnNetB0" / "EffiecientnNetB0_V6.keras"

# Create output directory (no subdirs)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 📦 Load Model ===
print(f"📦 Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# === 🧠 Parameters ===
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.80
class_names = ['leaf', 'no_leaf']  # Make this match training class order!
target_class = 'leaf'
target_class_idx = class_names.index(target_class)

# === 🔍 Process Images ===
all_images = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"🔍 Found {len(all_images)} images in raw directory.")

filtered_count = 0
skipped_count = 0
error_count = 0

for idx, fname in enumerate(all_images, 1):  # start from 1
    print(f"🔄 Processing image {idx} / {len(all_images)}")

    fpath = RAW_DIR / fname
    try:
        img = image.load_img(fpath, target_size=IMAGE_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = np.squeeze(model.predict(x))
        pred_class_idx = np.argmax(preds)
        confidence = preds[pred_class_idx]

        if pred_class_idx == target_class_idx and confidence >= CONFIDENCE_THRESHOLD:
            dest_path = OUTPUT_DIR / fname
            shutil.copy2(fpath, dest_path)
            filtered_count += 1
        else:
            print(f"⚠️ Skipped: predicted '{class_names[pred_class_idx]}' with confidence {confidence:.2f}")
            skipped_count += 1

    except UnidentifiedImageError:
        print(f"❌ Unreadable image: {fname}")
        error_count += 1
    except Exception as e:
        print(f"❌ Error processing {fname}: {str(e)}")
        error_count += 1

print("\n✅ Done!")
print(f"🟢 Saved {filtered_count} images to {OUTPUT_DIR}")
print(f"⚠️ Skipped {skipped_count} due to low confidence or wrong class")
print(f"❌ Encountered {error_count} unreadable or broken images")