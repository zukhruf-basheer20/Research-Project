import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
import matplotlib.pyplot as plt
import tarfile
import shutil

# === PATHS ===
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "tune"
LEAF_DIR = DATA_DIR / "leaf"
NO_LEAF_DIR = DATA_DIR / "no_leaf"
MODEL_NAME = "leaf_classifier_Ef_B4_V1"
MODEL_DIR = ROOT_DIR / "models / EfficientNetB4"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / f"{MODEL_NAME}_V1.keras"
WEIGHTS_DIR = ROOT_DIR / "trained_weights / EfficientNetB4"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_SAVE_PATH = WEIGHTS_DIR / f"{MODEL_NAME}_V1.weights.h5"
RESULTS_DIR = ROOT_DIR / "results / EfficientNetB4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_SAVE_PATH = RESULTS_DIR / f"{MODEL_NAME}_V1.png"
WEIGHTS_TAR = ROOT_DIR / "weights" / "plantnet" / "efficientnet_b4_weights_best_acc.tar"
TEMP_WEIGHTS_DIR = ROOT_DIR / "weights" / "plantnet" / "tmp_extracted_weights"
FINE_TUNE_DIR = ROOT_DIR / "fine_tune_classifier"
FINE_TUNE_DIR.mkdir(parents=True, exist_ok=True)  # For completeness

# === PARAMETERS ===
IMAGE_SIZE = (380, 380)
BATCH_SIZE = 16
EPOCHS = 25

# === SAFE EXTRACTION OF WEIGHTS ===
if not TEMP_WEIGHTS_DIR.exists():
    TEMP_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
with tarfile.open(WEIGHTS_TAR) as tar:
    tar.extractall(path=TEMP_WEIGHTS_DIR)

# Find the .h5 or best checkpoint in extracted files
weight_files = list(TEMP_WEIGHTS_DIR.glob('*.h5'))
if not weight_files:
    raise FileNotFoundError("No .h5 weights found in the tar archive!")
EFFICIENTNET_WEIGHTS_PATH = weight_files[0]  # Use the first found for simplicity

# === DATA AUGMENTATION ===
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2  # We'll split data here!
)

train_data = train_datagen.flow_from_directory(
    str(DATA_DIR),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    str(DATA_DIR),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# === MODEL DEFINITION & LOAD PRETRAINED WEIGHTS ===
base_model = EfficientNetB4(
    include_top=False,
    weights=None,  # We'll load custom weights below
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
base_model.load_weights(str(EFFICIENTNET_WEIGHTS_PATH), by_name=True, skip_mismatch=True)

base_model.trainable = True  # Full fine-tuning. Optionally, set layers you want to freeze!

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === CALLBACKS ===
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)
checkpoint_cb = callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
)

# === TRAINING ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint_cb]
)

# === SAVE WEIGHTS ===
model.save_weights(WEIGHTS_SAVE_PATH)
print(f"✅ Model weights saved to {WEIGHTS_SAVE_PATH}")

# === PLOTTING ===
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy', marker='o')
plt.plot(epochs_range, val_acc, label='Val Accuracy', marker='x')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Val Loss', marker='x')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(RESULTS_SAVE_PATH)
print(f"📈 Training curves saved to {RESULTS_SAVE_PATH}")

# === CLEANUP TEMP WEIGHTS DIR (OPTIONAL) ===
shutil.rmtree(TEMP_WEIGHTS_DIR)