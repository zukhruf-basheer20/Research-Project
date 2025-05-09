import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError
import seaborn as sns
import math

# === ✅ Safe ImageDataGenerator ===
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SafeImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, *args, **kwargs):
        generator = super().flow_from_directory(*args, **kwargs)
        generator._get_batches_of_transformed_samples = self._safe_batch(generator)
        return generator

    def _safe_batch(self, generator):
        original_get_batches = generator._get_batches_of_transformed_samples

        def safe_get_batches(index_array):
            batch = []
            labels = []
            for idx in index_array:
                try:
                    x, y = original_get_batches([idx])
                    batch.append(x[0])
                    labels.append(y[0])
                except Exception as e:
                    print(f"⚠️ Skipping unreadable image: {generator.filenames[idx]} ({e})")
            if batch:
                return np.array(batch), np.array(labels)
            else:
                raise ValueError("All images in this batch were invalid.")
        return safe_get_batches

# === 🔧 Path Setup ===
ROOT_DIR = Path(__file__).resolve().parents[1]  # -> Code/
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / "heracleum_classifier.h5"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 🖨️ Path Check
print(f"📁 Looking for train data in: {TRAIN_DIR}")
print(f"📁 Looking for val data in: {VAL_DIR}")

# === 🧠 Training Params ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# === 🧪 Data Generators (Safe) ===
train_gen = SafeImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=15,
    horizontal_flip=True
)

val_gen = SafeImageDataGenerator(
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

# === 🧠 Model ===
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 🚀 Train ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# === 💾 Save Model ===
model.save(str(MODEL_SAVE_PATH))
print(f"✅ Model saved to {MODEL_SAVE_PATH}")

# === 📊 Evaluate + Visualize ===
val_steps = math.ceil(val_data.samples / val_data.batch_size)
val_preds = model.predict(val_data, steps=val_steps)
y_true = val_data.classes
y_pred = np.argmax(val_preds, axis=1)

min_len = min(len(y_true), len(y_pred))
y_true = y_true[:min_len]
y_pred = y_pred[:min_len]

target_names = list(val_data.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred)

# === 📈 Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.show()

# === 📊 Precision / Recall / F1 Plot
metrics = ['precision', 'recall', 'f1-score']
plt.figure(figsize=(10, 6))

for metric in metrics:
    scores = [report[cls][metric] for cls in target_names]
    plt.plot(target_names, scores, marker='o', label=metric)

plt.ylim(0, 1.05)
plt.title("Classification Metrics per Class")
plt.xlabel("Class")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "metrics_per_class.png")
plt.show()