# 1️⃣ Install dependencies first:
# pip install pandas matplotlib basemap basemap-data-hires openpyxl

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import time

start_time = time.time()

# Load Excel data
file_path = "../CSV Files/EffNetB0_leaf_deep_features_with_metadata_V1.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# Define species
species_map = {
    "Heracleum_mantegazzianum_Sommier": "Heracleum Mantegazzianum Sommier",
    "Heracleum_sosnowskyi_Manden": "Heracleum Sosnowskyi Manden",
    "Heracleum_persicum_Desf": "Heracleum Persicum Desf"
}

def detect_species(filename):
    for key, label in species_map.items():
        if key in filename:
            return label
    return None

df["species_label"] = df["filename"].astype(str).apply(detect_species)
geo_df = df[df["species_label"].notnull()].dropna(subset=["latitude", "longitude"])

# Colors
fixed_colors = {
    "Heracleum Mantegazzianum Sommier": "red",
    "Heracleum Sosnowskyi Manden": "green",
    "Heracleum Persicum Desf": "blue"
}

# Create Basemap
plt.figure(figsize=(12,8))
m = Basemap(
    projection='merc',
    llcrnrlat=geo_df["latitude"].min() - 2,
    urcrnrlat=geo_df["latitude"].max() + 2,
    llcrnrlon=geo_df["longitude"].min() - 2,
    urcrnrlon=geo_df["longitude"].max() + 2,
    resolution='i'
)
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgray', lake_color='lightblue')

# Plot points
for species, color in fixed_colors.items():
    subset = geo_df[geo_df["species_label"] == species]
    if subset.empty:
        continue
    x, y = m(subset["longitude"].values, subset["latitude"].values)
    m.scatter(x, y, label=species, color=color, s=50, edgecolor='k', alpha=0.8)

plt.legend(title="Species")
plt.title("Heracleum Species Locations on Map")
plt.tight_layout()
plt.show()

end_time = time.time()
print(f"✅ Finished in {end_time - start_time:.2f} sec.")
