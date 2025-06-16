import os
import re
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers
import time

# === PATHS ===
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "data" / "Filtered_EffiecientnNetB0_V6"
MODEL_PATH = ROOT_DIR / "models" / "EffiecientnNetB0" / "EffiecientnNetB0_V6.keras"
OUTPUT_CSV = ROOT_DIR / "CSV Files" / "EffNetB0_leaf_deep_features_with_metadata_new_16June.csv"

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
scientific_name_list = []
taxon_rank_list = []
vernacular_name_list = []
habitat_list = []
recorded_by_list = []
identified_by_list = []
media_url_list = []

# === GBIF METADATA FETCHER ===
def get_gbif_metadata(gbif_id, fname, retries=2):
    url = f"https://api.gbif.org/v1/occurrence/{gbif_id}"
    attempt = 0

    while attempt <= retries:
        # print(f"🔍 Attempt {attempt + 1}/{retries + 1}: Fetching {url} (for file: {fname})")
        try:
            response = requests.get(url, timeout=10)
            # print(f"📡 Response status code: {response.status_code}")

            if response.status_code == 200:
                # print(f"✅ SUCCESS: Direct lookup for ID {gbif_id}")
                metadata = parse_metadata(response.json())
                # print(f"🌟 Extracted metadata for ID {gbif_id}: {metadata}")
                return metadata
            else:
                print(f"⚠️ WARNING: Direct lookup failed for ID {gbif_id} (Status: {response.status_code}) for file: {fname}")
        except Exception as e:
            print(f"💥 ERROR: Exception while fetching metadata for ID {gbif_id} (file: {fname}): {e}")

        attempt += 1
        if attempt <= retries:
            print(f"⏳ Retrying after 1 second...")
            time.sleep(1)

    print(f"❌ FINAL: Could not get exact metadata for ID {gbif_id} after {retries + 1} tries. File: {fname} | URL: {url}")
    return empty_metadata()

def parse_metadata(data):
    meta = {
        "latitude": data.get("decimalLatitude"),
        "longitude": data.get("decimalLongitude"),
        "country": data.get("country"),
        "event_date": data.get("eventDate"),
        "dataset_key": data.get("datasetKey"),
        "basis_of_record": data.get("basisOfRecord"),
        "scientific_name": data.get("scientificName"),
        "taxon_rank": data.get("taxonRank"),
        "vernacular_name": data.get("vernacularName"),
        "habitat": data.get("habitat"),
        "recorded_by": data.get("recordedBy"),
        "identified_by": data.get("identifiedBy"),
    }

    # Try to get media URL from extensions or media
    extensions = data.get("extensions") or {}
    multimedia = extensions.get("http://rs.gbif.org/terms/1.0/Multimedia")
    if multimedia and isinstance(multimedia, list):
        meta["media_url"] = multimedia[0].get("http://purl.org/dc/terms/identifier")
    else:
        media = data.get("media")
        if media and isinstance(media, list) and "identifier" in media[0]:
            meta["media_url"] = media[0]["identifier"]
        else:
            meta["media_url"] = None

    return meta

def empty_metadata():
    return dict.fromkeys([
        "latitude", "longitude", "country", "event_date",
        "dataset_key", "basis_of_record", "scientific_name",
        "taxon_rank", "vernacular_name", "habitat",
        "recorded_by", "identified_by", "media_url"
    ], None)

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

        vec = feature_extractor.predict(x, verbose=0)[0]
        features.append(vec)
        filenames.append(fname)

        # === Robust GBIF ID extraction ===
        parts = fname.rsplit("_", 2)
        gbif_id = None

        if len(parts) >= 2:
            last = parts[-1].split(".")[0]
            second_last = parts[-2]
            if last.isdigit() and second_last.isdigit():
                gbif_id = second_last  # e.g. with suffix
            elif last.isdigit():
                gbif_id = last  # no suffix
            else:
                gbif_id = None

        metadata = get_gbif_metadata(gbif_id, fname)

        latitude_list.append(metadata["latitude"])
        longitude_list.append(metadata["longitude"])
        country_list.append(metadata["country"])
        event_date_list.append(metadata["event_date"])
        dataset_key_list.append(metadata["dataset_key"])
        basis_of_record_list.append(metadata["basis_of_record"])
        scientific_name_list.append(metadata["scientific_name"])
        taxon_rank_list.append(metadata["taxon_rank"])
        vernacular_name_list.append(metadata["vernacular_name"])
        habitat_list.append(metadata["habitat"])
        recorded_by_list.append(metadata["recorded_by"])
        identified_by_list.append(metadata["identified_by"])
        media_url_list.append(metadata["media_url"])

        print(f"✅ Processed: {fname} ({i}/{len(image_files)})")

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

# === CREATE DATAFRAME ===
if features:
    feature_dim = len(features[0])
    columns = ['filename'] + [f'effnet_feature_{i+1}' for i in range(feature_dim)]

    df = pd.DataFrame(columns=columns)
    df['filename'] = filenames
    df.iloc[:, 1:1+feature_dim] = features

    # Add metadata
    df['latitude'] = latitude_list
    df['longitude'] = longitude_list
    df['country'] = country_list
    df['event_date'] = event_date_list
    df['dataset_key'] = dataset_key_list
    df['basis_of_record'] = basis_of_record_list
    df['scientific_name'] = scientific_name_list
    df['taxon_rank'] = taxon_rank_list
    df['vernacular_name'] = vernacular_name_list
    df['habitat'] = habitat_list
    df['recorded_by'] = recorded_by_list
    df['identified_by'] = identified_by_list
    df['media_url'] = media_url_list

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🟢 Features + metadata saved: {OUTPUT_CSV}")

else:
    print("⚠️ No features extracted — check your input folder!")
