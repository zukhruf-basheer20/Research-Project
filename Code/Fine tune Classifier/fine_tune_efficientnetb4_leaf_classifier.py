import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import shutil

# ==== Paths and Setup ====
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data' / 'tune'
MODEL_NAME = 'leaf_classifier'
MODEL_DIR = ROOT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / f'{MODEL_NAME}_V1.pt'
WEIGHTS_DIR = ROOT_DIR / 'trained_weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_SAVE_PATH = WEIGHTS_DIR / f'{MODEL_NAME}_V1.weights.pth'
RESULTS_DIR = ROOT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_SAVE_PATH = RESULTS_DIR / f'{MODEL_NAME}_V1.png'
PLANTNET_WEIGHTS_PATH = ROOT_DIR / 'Weights' / 'plantnet' / 'efficientnet_b4_weights_best_acc.tar'
FINE_TUNE_DIR = ROOT_DIR / 'fine_tune_classifier'
FINE_TUNE_DIR.mkdir(parents=True, exist_ok=True)

# ==== Device ====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

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
data_dir = DATA_DIR
# Split 20% of training data for validation
dataset = datasets.ImageFolder(data_dir)
class_names = dataset.classes
num_classes = len(class_names)
print(f"Classes found: {class_names}")

# Split train/val
from torch.utils.data import random_split, DataLoader
dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply transforms
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
dataloaders = {'train': train_loader, 'val': val_loader}

# ==== Model Setup (EfficientNet-B4) ====
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    raise ImportError("Install efficientnet_pytorch: pip install efficientnet_pytorch")

model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
state = torch.load(PLANTNET_WEIGHTS_PATH, map_location=device)
model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
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
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            for inputs, labels in dataloaders[phase]:
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

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'🏆 Best val Acc: {best_acc:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

# ==== Train ====
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

print("✨ All done! Go forth and classify.")
