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

