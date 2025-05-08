# File Tracker

## Dataset Tracking

### ✅ Raw Dataset (from GBIF)
- Folder: `data/raw/`
- Downloaded on: 2025-05-02
- Method: GBIF API
- Species: Quercus (oak) — Taxon Key: 2540970
- Images Downloaded: 50 (initial test batch)

### ⏳ Filtered Dataset
- Folder: `data/filtered/`
- Status: To be created
- Method: Binary classification using pre-trained model (ResNet50 or EfficientNetB0)

---

## Notes
- Expand download count after testing import + filter steps.
- Will separate into `leaf/` and `non_leaf/` if needed for training a classifier.
- Keep track of any corrupted/missing images during download phase.

code/
│
├── data/
│   ├── raw/              # GBIF downloaded images go here
│   ├── filtered/         # Images confirmed to contain leaves
│
├── notebooks/
│   └── 01_data_import.ipynb  # Your image import + GBIF download logic
├── python notebooks/
│   └── download_gbif_images.py
│
├── models/               # To store trained classifiers or segmentation models
│
├── results/              # Visual outputs like Grad-CAM, PCA
│
├── README.md             # Overview of the project
├── file_tracker.md       # Track of what’s imported, filtered, etc.
├── requirements.txt      # Python dependencies
