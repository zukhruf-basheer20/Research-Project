import requests

# === CONFIG ===
GBIF_ID = "4909190956"  # 👉 Replace this with your test ID

def parse_metadata(data):
    """Extracts useful fields from GBIF occurrence data"""
    return {
        "latitude": data.get("decimalLatitude"),
        "longitude": data.get("decimalLongitude"),
        "country": data.get("country"),
        "event_date": data.get("eventDate"),
        "dataset_key": data.get("datasetKey"),
        "basis_of_record": data.get("basisOfRecord"),
    }

def empty_metadata():
    """Fallback when nothing is found"""
    return {
        "latitude": None,
        "longitude": None,
        "country": None,
        "event_date": None,
        "dataset_key": None,
        "basis_of_record": None,
    }

def get_gbif_metadata(gbif_id):
    """Try occurrence/{id}, fallback to search?media={id}"""
    try:
        print(f"🔍 Trying direct lookup: /occurrence/{gbif_id}")
        url = f"https://api.gbif.org/v1/occurrence/{gbif_id}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            print(f"✅ Direct lookup succeeded")
            return parse_metadata(resp.json())

        print(f"⚠️ Direct lookup failed ({resp.status_code}), trying search?media=")
        search_url = f"https://api.gbif.org/v1/occurrence/search?media={gbif_id}"
        search_resp = requests.get(search_url, timeout=10)
        if search_resp.status_code == 200:
            results = search_resp.json().get("results", [])
            if results:
                print(f"✅ Fallback search succeeded, using first result")
                return parse_metadata(results[0])
            else:
                print(f"⚠️ Fallback search returned no results")
        else:
            print(f"❌ Fallback search failed ({search_resp.status_code})")

    except Exception as e:
        print(f"❌ Error during metadata fetch: {e}")

    print(f"⚠️ Returning empty metadata")
    return empty_metadata()


if __name__ == "__main__":
    metadata = get_gbif_metadata(GBIF_ID)
    print("\n🌍 === GBIF Metadata ===")
    for k, v in metadata.items():
        print(f"{k}: {v}")
