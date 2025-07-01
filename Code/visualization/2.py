# 1️⃣ Install dependencies first (in your terminal):
# pip install pandas matplotlib openpyxl

import pandas as pd
import matplotlib.pyplot as plt
import time

# Start timer
start_time = time.time()

print("🔹 [Checkpoint] Starting script...")

# 2️⃣ Load Excel data
file_path = "../CSV Files/EffNetB0_leaf_deep_features_with_metadata_V1.xlsx"

print(f"🔹 [Checkpoint] Loading Excel file: {file_path}")
df = pd.read_excel(file_path, sheet_name=0)

print("✅ [Checkpoint] File loaded.")
print(f"✅ Data shape: {df.shape}")

# 3️⃣ Define target keywords and readable names
species_map = {
    "Heracleum_mantegazzianum_Sommier": "Heracleum Mantegazzianum Sommier",
    "Heracleum_sosnowskyi_Manden": "Heracleum Sosnowskyi Manden",
    "Heracleum_persicum_Desf": "Heracleum Persicum Desf"
}

# 4️⃣ Assign species labels
def detect_species(filename):
    for key, label in species_map.items():
        if key in filename:
            return label
    return None

print("🔹 [Checkpoint] Tagging species...")
df["species_label"] = df["filename"].astype(str).apply(detect_species)

# 5️⃣ Filter rows with species
filtered_df = df[df["species_label"].notnull()]
print(f"✅ Rows with recognized species: {filtered_df.shape[0]}")

# 6️⃣ Drop rows without lat/lon
geo_df = filtered_df.dropna(subset=["latitude", "longitude"])
print(f"✅ Rows with valid coordinates: {geo_df.shape[0]}")

# 7️⃣ Define fixed colors
fixed_colors = {
    "Heracleum Mantegazzianum Sommier": "red",
    "Heracleum Sosnowskyi Manden": "green",
    "Heracleum Persicum Desf": "blue"
}

# 8️⃣ Plot
print("🔹 [Checkpoint] Generating scatter plot...")
plt.figure(figsize=(10,7))

for species, color in fixed_colors.items():
    subset = geo_df[geo_df["species_label"] == species]
    if subset.empty:
        continue
    plt.scatter(
        subset["longitude"],
        subset["latitude"],
        label=species,
        color=color,
        s=50,
        alpha=0.8,
        edgecolor='k'
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Heracleum Species Locations by Name")
plt.legend(title="Species", loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# Finish timer
end_time = time.time()
print(f"✅ [Checkpoint] Script finished successfully in {end_time - start_time:.2f} seconds.")
