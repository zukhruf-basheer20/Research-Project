import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError

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

# === 📊 Evaluate ===
val_preds = model.predict(val_data)
y_true = val_data.classes
y_pred = np.argmax(val_preds, axis=1)

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=val_data.class_indices.keys()))

print("🧾 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# === 📈 Plot Training Curves ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
plot_path = RESULTS_DIR / "training_plot.png"
plt.savefig(plot_path)
plt.show()
print(f"📊 Training curves saved to {plot_path}")