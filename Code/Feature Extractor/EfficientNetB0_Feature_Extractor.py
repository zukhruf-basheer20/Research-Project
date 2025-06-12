import os
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers

# === PATHS ===
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "data" / "Filtered_EffiecientnNetB0_V6"
MODEL_PATH = ROOT_DIR / "models" / "EffiecientnNetB0" / "EffiecientnNetB0_V6.keras"
OUTPUT_CSV = ROOT_DIR / "CSV Files" / "EffNetB0_leaf_deep_features.csv"

# === LOAD MODEL ===
print(f"📦 Loading model from: {MODEL_PATH}")
full_model = load_model(MODEL_PATH)

# === EXTRACT FEATURE LAYER FROM EFFICIENTNETB0 ===
# Grab the EfficientNetB0 layer from inside Sequential model
efficientnet_layer = full_model.layers[0]

# Apply GAP to get the feature vector (1280-d)
gap_output = layers.GlobalAveragePooling2D()(efficientnet_layer.output)

# Build a model that outputs the 1280-d feature vector
feature_extractor = Model(inputs=efficientnet_layer.input, outputs=gap_output)

# === CONFIG ===
IMAGE_SIZE = (224, 224)

# === INIT STORAGE ===
features = []
filenames = []

image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"🔍 Found {len(image_files)} images in: {INPUT_DIR}")

for i, fname in enumerate(image_files, 1):
    try:
        img_path = INPUT_DIR / fname
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict the feature vector
        vec = feature_extractor.predict(x, verbose=0)[0]
        features.append(vec)
        filenames.append(fname)

        print(f"✅ Extracted features from: {fname} ({i}/{len(image_files)})")

    except Exception as e:
        print(f"❌ Error processing {fname}: {str(e)}")

# === CREATE DATAFRAME ===
if features:
    feature_dim = len(features[0])
    column_names = ['filename'] + [f'effnet_feature_{i+1}' for i in range(feature_dim)]

    df = pd.DataFrame(columns=column_names)
    df['filename'] = filenames
    df.iloc[:, 1:] = features

    # === SAVE CSV ===
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🟢 Features saved to CSV: {OUTPUT_CSV}")
else:
    print("⚠️ No features extracted — check if input folder is empty or all images failed.")
