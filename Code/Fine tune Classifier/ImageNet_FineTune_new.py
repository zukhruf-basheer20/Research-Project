# imagenet_new_v1_train.py

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==== CONFIGURATION ====
MODEL_NAME = "Imagenet_new_V1"

MODEL_DIR = Path("models/new_imagenet")
WEIGHTS_DIR = Path("trained_weights")
RESULTS_DIR = Path("results/new_imagenet")
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# ==== CREATE FOLDERS IF NEEDED ====
for d in [MODEL_DIR, WEIGHTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==== DATA GENERATORS ====
train_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_data = train_gen.flow_from_directory(
    directory=str(TRAIN_DIR),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    directory=str(VAL_DIR),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ==== MODEL DEFINITION ====
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ==== CALLBACKS ====
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# ==== TRAIN ====
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ==== SAVE MODEL, WEIGHTS, AND PLOT ====
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.keras"
WEIGHTS_PATH = WEIGHTS_DIR / f"{MODEL_NAME}.weights.h5"
PLOT_PATH = RESULTS_DIR / f"{MODEL_NAME}_training_curves.png"

model.save(MODEL_PATH)
print(f"🟢 Model saved at: {MODEL_PATH}")

model.save_weights(WEIGHTS_PATH)
print(f"🟢 Weights saved at: {WEIGHTS_PATH}")

# ==== PLOT TRAINING CURVES ====
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
plt.savefig(PLOT_PATH)
print(f"🟢 Training curves saved at: {PLOT_PATH}")
