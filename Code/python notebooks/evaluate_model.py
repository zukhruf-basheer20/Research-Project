import numpy as np
from pathlib import Path
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# === ✅ Safe ImageDataGenerator ===
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

# === 🔧 Paths ===
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
VAL_DIR = DATA_DIR / "train"
MODEL_PATH = ROOT_DIR / "models" / "heracleum_classifier_best.keras"

# === 📦 Load Model ===
print(f"📦 Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# === 📂 Validation Data ===
val_gen = SafeImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

val_data = val_gen.flow_from_directory(
    directory=str(VAL_DIR),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# # === 🔍 Predict and Evaluate ===
# val_steps = math.ceil(val_data.samples / val_data.batch_size)
# val_preds = model.predict(val_data, steps=val_steps)
# y_true = val_data.classes
# y_pred = np.argmax(val_preds, axis=1)

# # Ensure same length
# min_len = min(len(y_true), len(y_pred))
# y_true = y_true[:min_len]
# y_pred = y_pred[:min_len]

# # === 📊 Report ===
# print("📊 Classification Report:")
# print(classification_report(y_true, y_pred, target_names=val_data.class_indices.keys()))

# print("🧾 Confusion Matrix:")
# print(confusion_matrix(y_true, y_pred))

# === 🔍 Predict and Evaluate ===
val_steps = math.ceil(val_data.samples / val_data.batch_size)
val_preds = model.predict(val_data, steps=val_steps)
y_true = val_data.classes
y_pred = np.argmax(val_preds, axis=1)

# Ensure same length
min_len = min(len(y_true), len(y_pred))
y_true = y_true[:min_len]
y_pred = y_pred[:min_len]

# === 📊 Report Text
target_names = list(val_data.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred)

# === 📈 Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(ROOT_DIR / "results" / "heracleum_classifier_best'keras'_confusionMatrix.png")
plt.show()

# === 📊 Plot Precision, Recall, F1-score
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
plt.savefig(ROOT_DIR / "results" / "heracleum_classifier_best'keras'_metricsPerClass.png")
plt.show()