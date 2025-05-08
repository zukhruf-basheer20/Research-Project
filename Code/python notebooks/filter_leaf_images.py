import os
import shutil
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf

# Paths
RAW_DIR = "../data/raw"
FILTERED_DIR = "../data/filtered"

# Create filtered directory if it doesn't exist
os.makedirs(FILTERED_DIR, exist_ok=True)

# Load EfficientNetB0
model = EfficientNetB0(weights="imagenet")
model = Model(inputs=model.input, outputs=model.output)

# Parameters
IMAGE_SIZE = (224, 224)
THRESHOLD_CLASSES = ["leaf", "plant", "tree", "flora"]

# Helper function to classify an image as leaf or not
def is_leaf(img_path):
    try:
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        decoded = decode_predictions(preds, top=3)[0]

        for _, label, prob in decoded:
            if any(keyword in label.lower() for keyword in THRESHOLD_CLASSES):
                return True
        return False
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

# Process flat image list in RAW_DIR
print(f"Scanning for images in {RAW_DIR}...")

all_images = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for fname in tqdm(all_images):
    src_path = os.path.join(RAW_DIR, fname)
    dest_path = os.path.join(FILTERED_DIR, fname)

    if is_leaf(src_path):
        shutil.copy2(src_path, dest_path)

print("Filtering complete. Leaf images saved in 'data/filtered/'.")