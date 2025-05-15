# I have used this sciprt to train EfficientNetB0 for better classification in the I trained on 400 data set and after that i have this model "heracleum_classifier_V2.keras"
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from PIL import UnidentifiedImageError

# === ✅ SafeImageDataGenerator ===
class SafeImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, *args, **kwargs):
        generator = super().flow_from_directory(*args, **kwargs)
        generator._get_batches_of_transformed_samples = self._safe_batch(generator)
        generator.skipped_images = []
        return generator

    def _safe_batch(self, generator):
        original_get_batches = generator._get_batches_of_transformed_samples

        def safe_get_batches(index_array):
            batch, labels = [], []
            for idx in index_array:
                try:
                    x, y = original_get_batches([idx])
                    batch.append(x[0])
                    labels.append(y[0])
                except Exception as e:
                    bad_file = generator.filenames[idx]
                    print(f"⚠️ Skipping unreadable image: {bad_file} ({e})")
                    generator.skipped_images.append(bad_file)
            if batch:
                return np.array(batch), np.array(labels)
            else:
                raise ValueError("All images in this batch were invalid.")
        return safe_get_batches

# === 📁 Paths ===
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / "heracleum_classifier_V2.keras"

# === 📸 Image Parameters ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# === 🔄 Data Generators (Upgraded Augmentation) ===
train_gen = SafeImageDataGenerator(
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

# === 🧠 Model Definition ===
print("🔧 Building base model...")
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

# === ⏹️ Callbacks ===
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

class SaveBestModelCallback(callbacks.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = Path(save_path)
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            if self.save_path.exists():
                self.save_path.unlink()
            self.model.save(self.save_path)
            print(f"✅ Model saved for epoch {epoch + 1}")
            print(f"🏆 New best model found at epoch {epoch + 1} (val_acc: {val_acc:.4f})")
        else:
            print(f"ℹ️ Epoch {epoch + 1} complete – no improvement (val_acc: {val_acc:.4f})")

save_best = SaveBestModelCallback(MODEL_SAVE_PATH)

# === 🚀 Training (no fine-tuning) ===
print("🚀 Starting training with frozen base model only...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stop, save_best]
)

# === 🧾 Summary of Skipped Images
skipped = set(train_data.skipped_images + val_data.skipped_images)
if skipped:
    print("\n🚫 Skipped Images Summary:")
    for img in skipped:
        print(f" - {img}")
    print(f"🧾 Total skipped: {len(skipped)}")
else:
    print("\n✅ No unreadable images were found.")
