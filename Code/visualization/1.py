# 1️⃣ Install dependencies first (in your terminal):
# pip install pandas umap-learn openpyxl matplotlib

import pandas as pd
import umap
import matplotlib.pyplot as plt
import time
import numpy as np

# Start timer
start_time = time.time()

print("🔹 [Checkpoint] Starting script...")

# 2️⃣ Load your Excel data
file_path = "../CSV Files/EffNetB0_leaf_deep_features_with_metadata_V1.xlsx"

print(f"🔹 [Checkpoint] Loading Excel file: {file_path}")

df = pd.read_excel(file_path, sheet_name=0)

# 3️⃣ Sanity check
print("✅ [Checkpoint] Excel file loaded successfully.")
print(f"✅ Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("✅ Columns preview:", list(df.columns[:5]), "...")

# 4️⃣ Prepare your feature matrix
print("🔹 [Checkpoint] Selecting numeric columns only...")

numeric_cols = df.select_dtypes(include=['number']).columns
print("✅ Numeric columns detected:", list(numeric_cols[:5]), "...")
print(f"✅ Total numeric columns: {len(numeric_cols)}")

X = df[numeric_cols].values
print(f"✅ Feature matrix shape: {X.shape}")

# 4️⃣.1 Optional: Check for NaNs
nan_count = np.isnan(X).sum()
print(f"✅ Checking for NaNs: {nan_count}")
if nan_count > 0:
    print("⚠️ Warning: NaNs detected! Filling with column means.")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    print("✅ NaNs filled.")

# 5️⃣ Create and fit UMAP
print("🔹 [Checkpoint] Running UMAP dimensionality reduction...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
embedding = reducer.fit_transform(X)

print("✅ [Checkpoint] UMAP fitting complete.")
print(f"✅ Embedding shape: {embedding.shape}")

# 6️⃣ Plotting
print("🔹 [Checkpoint] Generating plot...")
plt.figure(figsize=(10,7))
plt.scatter(embedding[:,0], embedding[:,1], s=10, cmap='Spectral')
plt.title("UMAP Projection of Geopolitical Feature Vectors")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# Finish timer
end_time = time.time()
print(f"✅ [Checkpoint] Script finished successfully in {end_time - start_time:.2f} seconds.")
