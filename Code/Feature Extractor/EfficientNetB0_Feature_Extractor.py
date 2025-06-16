import os
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers

# === PATHS ===
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "data" / "Filtered_EffiecientnNetB0_V6"
MODEL_PATH = ROOT_DIR / "models" / "EffiecientnNetB0" / "EffiecientnNetB0_V6.keras"
OUTPUT_CSV = ROOT_DIR / "CSV Files" / "EffNetB0_leaf_deep_features_with_metadata.csv"

# === LOAD MODEL ===
print(f"📦 Loading model from: {MODEL_PATH}")
full_model = load_model(MODEL_PATH)

# === EXTRACT FEATURE LAYER FROM EFFICIENTNETB0 ===
efficientnet_layer = full_model.layers[0]
gap_output = layers.GlobalAveragePooling2D()(efficientnet_layer.output)
feature_extractor = Model(inputs=efficientnet_layer.input, outputs=gap_output)

# === CONFIG ===
IMAGE_SIZE = (224, 224)

# === INIT STORAGE ===
features = []
filenames = []

# Metadata storage
latitude_list = []
longitude_list = []
country_list = []
event_date_list = []
dataset_key_list = []
basis_of_record_list = []

# === GBIF METADATA FETCHER ===
def get_gbif_metadata(gbif_id):
    try:
        url = f"https://api.gbif.org/v1/occurrence/{gbif_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "latitude": data.get("decimalLatitude"),
                "longitude": data.get("decimalLongitude"),
                "country": data.get("country"),
                "event_date": data.get("eventDate"),
                "dataset_key": data.get("datasetKey"),
                "basis_of_record": data.get("basisOfRecord")
            }
    except Exception as e:
        print(f"⚠️ Failed to fetch metadata for GBIF ID {gbif_id}: {e}")
    return {
        "latitude": None,
        "longitude": None,
        "country": None,
        "event_date": None,
        "dataset_key": None,
        "basis_of_record": None
    }

# === FIND IMAGES ===
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"🔍 Found {len(image_files)} images in: {INPUT_DIR}")

# === PROCESS IMAGES ===
for i, fname in enumerate(image_files, 1):
    try:
        img_path = INPUT_DIR / fname
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract deep features
        vec = feature_extractor.predict(x, verbose=0)[0]
        features.append(vec)
        filenames.append(fname)

        # === Parse filename to extract GBIF ID ===
        parts = fname.rsplit("_", 2)
        gbif_id = parts[-2] if len(parts) >= 2 else None

        # === Get metadata from GBIF API ===
        metadata = get_gbif_metadata(gbif_id)

        latitude_list.append(metadata["latitude"])
        longitude_list.append(metadata["longitude"])
        country_list.append(metadata["country"])
        event_date_list.append(metadata["event_date"])
        dataset_key_list.append(metadata["dataset_key"])
        basis_of_record_list.append(metadata["basis_of_record"])

        print(f"✅ Extracted features and metadata from: {fname} ({i}/{len(image_files)})")

    except Exception as e:
        print(f"❌ Error processing {fname}: {str(e)}")

# === CREATE DATAFRAME ===
if features:
    feature_dim = len(features[0])
    column_names = ['filename'] + [f'effnet_feature_{i+1}' for i in range(feature_dim)]

    df = pd.DataFrame(columns=column_names)
    df['filename'] = filenames
    df.iloc[:, 1:1+feature_dim] = features

    # Add metadata columns
    df['latitude'] = latitude_list
    df['longitude'] = longitude_list
    df['country'] = country_list
    df['event_date'] = event_date_list
    df['dataset_key'] = dataset_key_list
    df['basis_of_record'] = basis_of_record_list

    # === SAVE CSV ===
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🟢 Features + metadata saved to CSV: {OUTPUT_CSV}")
else:
    print("⚠️ No features extracted — check if input folder is empty or images failed.")