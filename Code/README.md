code/
│
├── data/
│   ├── raw/              # GBIF downloaded images go here
│   ├── filtered/         # Images confirmed to contain leaves
│
├── notebooks/
│   └── 01_data_import.ipynb  # Your image import + GBIF download logic
│
├── models/               # To store trained classifiers or segmentation models
│
├── results/              # Visual outputs like Grad-CAM, PCA
│
├── README.md             # Overview of the project
├── file_tracker.md       # Track of what’s imported, filtered, etc.
├── requirements.txt      # Python dependencies