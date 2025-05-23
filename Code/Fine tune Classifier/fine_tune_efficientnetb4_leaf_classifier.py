import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==== Directory Setup ====
def make_dir(path):
    if not path.exists():
        path.mkdir(parents=True)
        print(f"📁 Created directory: {path}")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data' / 'tune'
MODEL_NAME = 'EfficientNetB4_V1'
MODEL_DIR = ROOT_DIR / 'models' / 'EfficientNetB4'
make_dir(MODEL_DIR)
MODEL_SAVE_PATH = MODEL_DIR / f'{MODEL_NAME}.pt'
WEIGHTS_DIR = ROOT_DIR / 'trained_weights'
make_dir(WEIGHTS_DIR)
WEIGHTS_SAVE_PATH = WEIGHTS_DIR / f'{MODEL_NAME}.weights.pth'
RESULTS_DIR = ROOT_DIR / 'results'
make_dir(RESULTS_DIR)
RESULTS_SAVE_PATH = RESULTS_DIR / f'{MODEL_NAME}.png'
PLANTNET_WEIGHTS_PATH = ROOT_DIR / 'weights' / 'plantnet' / 'efficientnet_b4_weights_best_acc.tar'
FINE_TUNE_DIR = ROOT_DIR / 'fine_tune_classifier'
make_dir(FINE_TUNE_DIR)

# ==== Device ====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

# ==== Corrupt Image Checker ====
def check_corrupted_images(data_dir):
    print("🔍 Scanning for corrupted images...")
    corrupted = []
    for folder in ["leaf", "no_leaf"]:
        folder_path = data_dir / folder
        if not folder_path.exists():
            print(f"⚠️ Warning: {folder_path} does not exist!")
            continue
        for img_file in folder_path.rglob("*"):
            if img_file.is_file():
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                except Exception as e:
                    print(f"❌ Corrupted: {img_file} ({e})")
                    corrupted.append(img_file)
    if not corrupted:
        print("✅ No corrupted images found in leaf/no_leaf.")
    else:
        print(f"🚨 {len(corrupted)} corrupted images found! (see above)")
    print("----\n")

check_corrupted_images(DATA_DIR)

# ==== Data Transforms ====
IMAGE_SIZE = 380
BATCH_SIZE = 16
EPOCHS = 20

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ==== Datasets and Dataloaders ====
print("🧩 Loading dataset...")
dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes
num_classes = len(class_names)
print(f"🌱 Classes found: {class_names}")

from torch.utils.data import random_split, DataLoader
dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dataloaders = {'train': train_loader, 'val': val_loader}

print("✅ Data loaders ready.")

# ==== Model Setup (EfficientNet-B4) ====
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    raise ImportError("Install efficientnet_pytorch: pip install efficientnet_pytorch")

model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
state = torch.load(PLANTNET_WEIGHTS_PATH, map_location=device)
print(f"Checkpoint keys: {state.keys()}")
model.load_state_dict(state["model"], strict=False)
print("✅ Loaded PlantNet EfficientNet-B4 weights.")

# Adapt final FC layer for binary classification
if num_classes == 2:
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 1)
    criterion = nn.BCEWithLogitsLoss()
else:
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    criterion = nn.CrossEntropyLoss()
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ==== Training Loop ====
def train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCHS):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f'\n🌙 Epoch {epoch+1}/{num_epochs} — Let the learning begin!')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            # Print progress every 10 batches
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).float() if num_classes == 2 else labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if num_classes == 2:
                        outputs = outputs.squeeze(1)
                        preds = torch.sigmoid(outputs) > 0.5
                        loss = criterion(outputs, labels)
                    else:
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                if num_classes == 2:
                    running_corrects += torch.sum(preds == labels.bool()).item()
                else:
                    running_corrects += torch.sum(preds == labels).item()
                running_total += labels.size(0)

                if batch_idx % 10 == 0:
                    print(f"   [{phase}] Batch {batch_idx+1} | Loss: {loss.item():.4f}")

            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects / running_total

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                scheduler.step()
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'\n🏆 Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

# ==== Train ====
print("🚦 Starting training loop...")
model, train_acc, val_acc, train_loss, val_loss = train_model(
    model, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)

# ==== Save Model ====
torch.save(model, MODEL_SAVE_PATH)
torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
print(f"✅ Model weights saved to {WEIGHTS_SAVE_PATH}")

# ==== Plot Training Curves ====
epochs_range = range(1, len(train_acc) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Train Acc', marker='o')
plt.plot(epochs_range, val_acc, label='Val Acc', marker='x')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Train Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Val Loss', marker='x')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(RESULTS_SAVE_PATH)
print(f"📈 Training plot saved to {RESULTS_SAVE_PATH}")

print("✨ All done! Go forth and classify with swagger.")