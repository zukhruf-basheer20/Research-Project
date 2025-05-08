import requests

def get_image_count(scientific_name):
    res = requests.get("https://api.gbif.org/v1/species/match", params={"name": scientific_name})
    res.raise_for_status()
    taxon_key = res.json().get("usageKey")

    if not taxon_key:
        print(f"Could not find taxonKey for {scientific_name}")
        return 0

    count_res = requests.get("https://api.gbif.org/v1/occurrence/search", params={
        "taxonKey": taxon_key,
        "mediaType": "StillImage",
        "limit": 0
    })
    count_res.raise_for_status()
    count = count_res.json().get("count", 0)
    print(f"{scientific_name} → {count} images available on GBIF")
    return count

# Heracleum species list
species = [
    "Heracleum mantegazzianum Sommier",
    "Heracleum sosnowskyi Manden.",
    "Heracleum persicum Desf."
]

total = 0
for s in species:
    total += get_image_count(s)

print(f"\nTOTAL images across all species: {total}")
