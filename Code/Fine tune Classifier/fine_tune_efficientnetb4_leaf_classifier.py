import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import ImageFile, Image
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Directory setup (adapt paths as needed)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data' / 'tune'
MODEL_NAME = 'EfficientNetB4_V7'
MODEL_DIR = ROOT_DIR / 'models' / 'EfficientNetB4'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / f'{MODEL_NAME}.pt'
WEIGHTS_DIR = ROOT_DIR / 'trained_weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_SAVE_PATH = WEIGHTS_DIR / f'{MODEL_NAME}.weights.pth'
RESULTS_DIR = ROOT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_SAVE_PATH = RESULTS_DIR / f'{MODEL_NAME}.png'
PLANTNET_WEIGHTS_PATH = ROOT_DIR / 'weights' / 'plantnet' / 'efficientnet_b4_weights_best_acc.tar'
FINE_TUNE_DIR = ROOT_DIR / 'fine_tune_classifier'
FINE_TUNE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

# Data transforms
IMAGE_SIZE = 380
BATCH_SIZE = 8
EPOCHS = 30

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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

# Datasets and dataloaders
dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes
num_classes = len(class_names)

from torch.utils.data import random_split, DataLoader
dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dataloaders = {'train': train_loader, 'val': val_loader}

# Model: EfficientNetB4 with pre-trained PlantNet weights
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    raise ImportError("Install efficientnet_pytorch: pip install efficientnet_pytorch")

model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
state = torch.load(PLANTNET_WEIGHTS_PATH, map_location=device)
model.load_state_dict(state["model"], strict=False)

# Freeze all layers, then unfreeze last 2 blocks + classifier
for param in model.parameters():
    param.requires_grad = False
for param in model._blocks[-2].parameters():
    param.requires_grad = True
for param in model._blocks[-1].parameters():
    param.requires_grad = True
for param in model._fc.parameters():
    param.requires_grad = True

# Set dropout
if hasattr(model, "_dropout"):
    model._dropout.p = 0.2

# Adapt FC for binary
if num_classes == 2:
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 1)
    criterion = nn.BCEWithLogitsLoss()
else:
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    criterion = nn.CrossEntropyLoss()
model = model.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-5)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCHS, patience=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    early_stopper = EarlyStopping(patience=patience)
    for epoch in range(num_epochs):
        print(f'\n🌙 Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_total = 0

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

            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects / running_total

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                scheduler.step()
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                early_stopper(epoch_loss)
                if early_stopper.early_stop:
                    print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
                    model.load_state_dict(best_model_wts)
                    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'\n🏆 Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

print("🚦 Starting training loop...")
model, train_acc, val_acc, train_loss, val_loss = train_model(
    model, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS, patience=10)

torch.save(model, MODEL_SAVE_PATH)
torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
print(f"✅ Model weights saved to {WEIGHTS_SAVE_PATH}")

# Plot training curves
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

print("==== Final Training Results ====")
print(f"Train Acc: {train_acc[-1]:.4f}")
print(f"Train Loss: {train_loss[-1]:.4f}")
print(f"Val Acc: {val_acc[-1]:.4f}")
print(f"Val Loss: {val_loss[-1]:.4f}")

print("✨ All done! Go forth and classify with swagger.")