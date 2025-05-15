# This Script can be used to fetch the data set from GBIF just from one perticular type 
import os
import requests
import time
from pathlib import Path
from collections import defaultdict
from threading import Thread

# === CONFIG ===
species_info = {
    "Heracleum sosnowskyi Manden.": "Heracleum_sosnowskyi_Manden",
}

RAW_DIR = Path("../data/raw")  # Adjusted for expected project structure
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Tracking counts
download_counts = defaultdict(int)
total_to_download = defaultdict(int)

def fetch_occurrences(scientific_name, limit=300):
    base_url = "https://api.gbif.org/v1/occurrence/search"

    # First get the actual count of relevant records
    count_resp = requests.get(base_url, params={
        "scientificName": scientific_name,
        "mediaType": "StillImage",
        "limit": 0
    })
    total = count_resp.json().get("count", 0)
    print(f"\nFound {total} potential records with images for: {scientific_name}")
    total_to_download[scientific_name] = total

    offset = 0
    occurrences = []

    while offset < total:
        params = {
            "scientificName": scientific_name,
            "mediaType": "StillImage",
            "limit": limit,
            "offset": offset
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code} for {scientific_name} at offset {offset}")
            break

        results = response.json().get("results", [])
        if not results:
            break

        occurrences.extend(results)
        print(f"➤ Fetched {len(results)} records (offset={offset})")
        offset += limit
        time.sleep(1)

    print(f"Done fetching {len(occurrences)} records for {scientific_name}")
    return occurrences

def download_images(sci_name, pretty_name, occurrences):
    for occ in occurrences:
        gbif_id = occ.get("key")
        media = occ.get("media", [])

        for idx, item in enumerate(media):
            img_url = item.get("identifier")
            if not img_url:
                continue

            suffix = f"_{idx+1}" if len(media) > 1 else ""
            filename = f"{pretty_name}_{gbif_id}{suffix}.jpg"
            filepath = RAW_DIR / filename

            if filepath.exists():
                continue

            try:
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    download_counts[sci_name] += 1
                else:
                    print(f"Skipped {img_url} — HTTP {response.status_code}")
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")

def progress_report():
    while True:
        time.sleep(20)
        print("\n📦 Download Progress:")
        total = 0
        for sci_name in species_info:
            count = download_counts[sci_name]
            max_count = total_to_download[sci_name]
            print(f"  {sci_name}: {count} / {max_count} downloaded")
            total += count
        print(f"Total downloaded so far: {total} images\n")

def main():
    print("Fetching image metadata from GBIF...")
    all_occurrences = {}

    for sci_name, pretty_name in species_info.items():
        occs = fetch_occurrences(sci_name)
        all_occurrences[sci_name] = occs

    # Show summary before download
    print("\nSummary of images to attempt:")
    grand_total = 0
    for sci_name, total in total_to_download.items():
        print(f"  {sci_name}: {total} image entries")
        grand_total += total
    print(f"Total expected image entries: {grand_total}\n")

    # Start live progress tracking in background
    Thread(target=progress_report, daemon=True).start()

    # Download images
    for sci_name, pretty_name in species_info.items():
        print(f"\nStarting download for: {sci_name}")
        download_images(sci_name, pretty_name, all_occurrences[sci_name])

    print("\nAll downloads complete.")

if __name__ == "__main__":
    main()