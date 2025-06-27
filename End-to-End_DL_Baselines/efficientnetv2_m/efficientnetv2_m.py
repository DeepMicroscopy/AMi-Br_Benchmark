import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import logging
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import timm
import numpy as np

# Logging setup
log_file = "amibr_efficientnetv2_m.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Model
class BinaryEfficientNetV2M(nn.Module):
    def __init__(self):
        super(BinaryEfficientNetV2M, self).__init__()
        self.model = timm.create_model('efficientnetv2_m', pretrained=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)

    def forward(self, x):
        return self.model(x)

# Training configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
num_epochs = 100
early_stop_patience = 15
classes = ['Atypical', 'Normal']
criterion = nn.BCEWithLogitsLoss()
fold_accuracies = []

# Data loading
df = pd.read_csv('/data/MELBA-AmiBr/Datasets_Stratified/AMi-Br/amibr_train_split.csv')
df['label'] = df['final_label'].map({'Atypical': 0, 'Normal': 1})
df['image_path'] = '/data/MELBA-AmiBr/Datasets_Stratified/AMi-Br/Train/' + df['final_label'] + '/' + df['dataset_uid']

images = df['image_path'].tolist()
labels = df['label'].tolist()
groups = df['slide'].tolist()

strat_group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation
for fold, (train_idx, val_idx) in enumerate(strat_group_kfold.split(images, labels, groups)):
    logger.info(f"--- Starting Fold {fold + 1} ---")

    # Split data
    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
    val_dataset = CustomDataset(val_images, val_labels, transform=val_transform)

    # Handle class imbalance
    unique_train_labels, counts = np.unique(train_labels, return_counts=True)
    class_counts = {label: count for label, count in zip(unique_train_labels, counts)}
    for i in range(len(classes)):
        if i not in class_counts:
            class_counts[i] = 0

    class_weights = {label: 1.0 / count if count > 0 else 0.0 for label, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Model, optimizer, scheduler
    model = BinaryEfficientNetV2M().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # We want to maximize validation accuracy
        factor=0.5,           # Reduce LR by half
        patience=3,           # Wait 3 epochs before reducing
        min_lr=1e-7,          # Don't go below this LR
        verbose=True          # Print when LR changes
    )

    best_val_balanced_accuracy = 0.0
    best_model_path = f'amibr_efficientnetv2m_fold_{fold + 1}_best.pth'
    epochs_no_improve = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds_logits, train_targets = [], []

        for images_batch, labels_batch in tqdm(train_loader, desc=f"Fold {fold + 1} - Epoch {epoch + 1} Training"):
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds_logits.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels_batch.detach().cpu().numpy())

        train_preds = (torch.sigmoid(torch.tensor(train_preds_logits)) > 0.5).numpy().astype(int)
        train_balanced_accuracy = balanced_accuracy_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_logits, val_targets = [], []

        with torch.no_grad():
            for images_batch, labels_batch in tqdm(val_loader, desc=f"Fold {fold + 1} - Epoch {epoch + 1} Validation"):
                images_batch = images_batch.to(device)
                labels_batch = labels_batch.float().unsqueeze(1).to(device)

                outputs = model(images_batch)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                val_preds_logits.extend(outputs.detach().cpu().numpy())
                val_targets.extend(labels_batch.detach().cpu().numpy())


        val_preds = (torch.sigmoid(torch.tensor(val_preds_logits)) > 0.5).numpy().astype(int)
        val_balanced_accuracy = balanced_accuracy_score(val_targets, val_preds)

        logger.info(
            f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Balanced Accuracy: {train_balanced_accuracy:.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            f"Val Balanced Accuracy: {val_balanced_accuracy:.4f}"
        )

        # Early stopping
        if val_balanced_accuracy > best_val_balanced_accuracy:
            best_val_balanced_accuracy = val_balanced_accuracy
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            logger.info(f"New best validation balanced accuracy: {best_val_balanced_accuracy:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            # logger.info(f"Validation balanced accuracy did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1} for Fold {fold + 1}.")
                break

        scheduler.step(val_balanced_accuracy)

    logger.info(f"Fold {fold + 1} - Best Validation Balanced Accuracy: {best_val_balanced_accuracy:.4f}")
    fold_accuracies.append(best_val_balanced_accuracy)
    torch.cuda.empty_cache()

# Final results
average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
logger.info("--- Training Complete ---")
logger.info(f"Validation Balanced Accuracies per fold: {[f'{acc:.4f}' for acc in fold_accuracies]}")
logger.info(f"Average Validation Balanced Accuracy across all folds: {average_accuracy:.4f}")
