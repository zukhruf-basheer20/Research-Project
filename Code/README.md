### 📁 Project Structure or Folder Structure
```sh
Code/
│
├── data/
│   ├── raw/                                    # GBIF downloaded images go here
│   ├── filtered/                               # Images confirmed to contain leaves
│   ├── train/
│   │   ├── Heracleum_mantegazzianum/
│   │   ├── Heracleum_persicum/
│   │   ├── Heracleum_sosnowskyi/
│   ├── val/
│   │   ├── Heracleum_mantegazzianum/
│   │   ├── Heracleum_persicum/
│   │   ├── Heracleum_sosnowskyi/
│
├── python notebooks/
│   ├── check_images.py                         # Sanity check for image integrity
│   ├── download_gbif_images_singular.py        # Script to download images from GBIF any one plant type
│   ├── download_gbif_images.py                 # Script to download images from GBIF
│   ├── filter_leaf_images.py                   # Filters images that actually contain leaves
│   ├── split_dataset.py                        # Train/val split
│   ├── train_model.py                          # Model training script
│
├── models/                                     # To store trained classifiers or segmentation models
│   ├── heracleum_classifier_best.keras
│
├── results/                                    # Visual outputs like Grad-CAM, PCA
│
├── README.md                                   # Overview of the project (you're reading it!)
├── requirements.txt                            # Python dependencies
```
To grip count of images
```
ls | grep "Heracleum_mantegazzianum_Sommier" | wc -l
ls | grep "Heracleum_sosnowskyi_Manden" | wc -l
ls | grep "Heracleum_persicum_Desf" | wc -l
```
To delete files
```
ls | grep "Heracleum_mantegazzianum_Sommier" | xargs rm
```