# 1️⃣ Install dependencies first (in your terminal):
# pip install pandas umap-learn matplotlib openpyxl

import pandas as pd
import umap
import matplotlib.pyplot as plt
import time
import numpy as np

# Start timer
start_time = time.time()

print("🔹 [Checkpoint] Starting script...")

# Load Excel data
file_path = "../CSV Files/EffNetB0_leaf_deep_features_with_metadata_V1.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

print("✅ File loaded.")
print(f"✅ Data shape: {df.shape}")

# Define species mapping
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

# Assign species label
df["species_label"] = df["filename"].astype(str).apply(detect_species)

# Filter only rows with a label
labeled_df = df[df["species_label"].notnull()]
print(f"✅ Rows with recognized species: {labeled_df.shape[0]}")

# Select numeric columns only (the feature vectors)
numeric_cols = labeled_df.select_dtypes(include=['number']).columns
# Exclude lat/lon if you don't want them as features:
numeric_cols = [col for col in numeric_cols if col not in ["latitude", "longitude"]]

print(f"✅ Total numeric feature columns: {len(numeric_cols)}")
print("✅ First 5 numeric columns:", numeric_cols[:5])

X = labeled_df[numeric_cols].values
print(f"✅ Feature matrix shape: {X.shape}")

# Handle NaNs
nan_count = np.isnan(X).sum()
if nan_count > 0:
    print("⚠️ NaNs detected, filling with column means...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    print("✅ NaNs filled.")

# UMAP projection
print("🔹 Running UMAP...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42,
    n_jobs=-1
)
embedding = reducer.fit_transform(X)
print("✅ UMAP embedding complete.")

# Plotting
plt.figure(figsize=(10,7))
species_list = labeled_df["species_label"].unique()
colors = ["red", "green", "blue"]
color_map = {species: colors[i] for i, species in enumerate(species_list)}

for species in species_list:
    subset = embedding[labeled_df["species_label"] == species]
    plt.scatter(
        subset[:,0],
        subset[:,1],
        label=species,
        color=color_map[species],
        s=30,
        alpha=0.7,
        edgecolor='k'
    )

plt.title("UMAP Projection of Feature Vectors")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Species")
plt.grid(True)
plt.tight_layout()
plt.show()

end_time = time.time()
print(f"✅ Finished in {end_time - start_time:.2f} sec.")
