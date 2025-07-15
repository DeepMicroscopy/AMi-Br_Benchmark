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
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

# Logging setup
log_file = "virchow2_linear_probe_training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face login
login(token="Your HuggingFace Token Here")

# Load Virchow2 feature extractor
virchow_model = timm.create_model(
    "hf-hub:paige-ai/Virchow2", 
    pretrained=True,
    mlp_layer=SwiGLUPacked, 
    act_layer=torch.nn.SiLU
)
virchow_model.eval().to(device)
virchow_config = resolve_data_config(virchow_model.pretrained_cfg, model=virchow_model)
virchow_transform = create_transform(**virchow_config)

# Classifier head
class VirchowBinaryClassifier(nn.Module):
    def __init__(self):
        super(VirchowBinaryClassifier, self).__init__()
        self.classifier = nn.Linear(2560, 1)

    def forward(self, x):
        return self.classifier(x)

# Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.embeddings = self._extract_embeddings()

    def _extract_embeddings(self):
        embeddings = []
        for img_path in tqdm(self.image_paths, desc="Extracting embeddings"):
            image = Image.open(img_path).convert("RGB")
            image_tensor = virchow_transform(image).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                output = virchow_model(image_tensor)
                class_token = output[:, 0]
                patch_tokens = output[:, 5:]
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1).squeeze(0).to(torch.float32)
                embeddings.append(embedding.cpu())
        return embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return embedding, label

# Load dataset
df = pd.read_csv('/data/MELBA-AmiBr/Datasets_Stratified/AMi-Br/amibr_train_split.csv')
df['label'] = df['final_label'].map({'Atypical': 0, 'Normal': 1})
df['image_path'] = '/data/MELBA-AmiBr/Datasets_Stratified/AMi-Br/Train/' + df['final_label'] + '/' + df['dataset_uid']

images = df['image_path'].tolist()
labels = df['label'].tolist()
groups = df['slide'].tolist()

# Hyperparameters
batch_size = 8
num_epochs = 100
early_stop_patience = 15
criterion = nn.BCEWithLogitsLoss()
fold_accuracies = []
strat_group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(strat_group_kfold.split(images, labels, groups)):
    logger.info(f"Starting Fold {fold + 1}")

    # Initialize model per fold
    model = VirchowBinaryClassifier().to(device)

    # Build datasets
    train_dataset = EmbeddingDataset([images[i] for i in train_idx], [labels[i] for i in train_idx])
    val_dataset = EmbeddingDataset([images[i] for i in val_idx], [labels[i] for i in val_idx])

    # Compute class weights from train split only
    train_labels = [labels[i] for i in train_idx]
    class_counts = [train_labels.count(i) for i in range(2)]
    class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [class_weights[train_labels[i]] for i in range(len(train_labels))]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # We want to maximize validation accuracy
        factor=0.5,           # Reduce LR by half
        patience=3,           # Wait 3 epochs before reducing
        min_lr=1e-7,          # Don't go below this LR
        verbose=True          # Print when LR changes
    )

    best_val_acc = 0.0
    best_model_path = f'virchow2_linear_probe_fold_{fold + 1}_best.pth'
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for embeddings, labels_batch in tqdm(train_loader, desc=f"Fold {fold + 1} - Epoch {epoch + 1} Training"):
            embeddings, labels_batch = embeddings.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
            train_targets.extend(labels_batch.cpu().numpy())

        train_bal_acc = balanced_accuracy_score(train_targets, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for embeddings, labels_batch in tqdm(val_loader, desc=f"Fold {fold + 1} - Epoch {epoch + 1} Validation"):
                embeddings, labels_batch = embeddings.to(device), labels_batch.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                val_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
                val_targets.extend(labels_batch.cpu().numpy())

        val_bal_acc = balanced_accuracy_score(val_targets, val_preds)

        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            torch.save(model, best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        scheduler.step(val_bal_acc)

        logger.info(
            f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Bal Acc: {train_bal_acc:.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            f"Val Bal Acc: {val_bal_acc:.4f}"
        )

    logger.info(f"Fold {fold + 1} - Best Validation Balanced Accuracy: {best_val_acc:.4f}")
    fold_accuracies.append(best_val_acc)
    torch.cuda.empty_cache()

# Final summary
avg_acc = sum(fold_accuracies) / len(fold_accuracies)
logger.info(f"Average Validation Balanced Accuracy across folds: {avg_acc:.4f}")