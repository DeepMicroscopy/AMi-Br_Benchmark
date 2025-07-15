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

# Logging setup
log_file = "hoptimus0_linear_probe.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Login & load model
login(token="Your HuggingFace Token Here")
hoptimus_model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
hoptimus_model.eval().to(device)

# Transform
hoptimus_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517))
])

# Embedding extractor
def extract_embedding(img_path):
    image = Image.open(img_path).convert("RGB").resize((224, 224))
    tensor = hoptimus_transform(image).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        features = hoptimus_model(tensor)
    return features.squeeze(0).cpu()

# Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.embeddings = [extract_embedding(p) for p in tqdm(image_paths, desc="Extracting embeddings")]
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

# Classifier
class HoptimusBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # No sigmoid
        )

    def forward(self, x):
        return self.classifier(x)

# Load CSV
df = pd.read_csv('/data/MELBA-AmiBr/Datasets_Stratified/AMi-Br/amibr_train_split.csv')
df['label'] = df['final_label'].map({'Atypical': 0, 'Normal': 1})
df['image_path'] = '/data/MELBA-AmiBr/Datasets_Stratified/AMi-Br/Train/' + df['final_label'] + '/' + df['dataset_uid']

images = df['image_path'].tolist()
labels = df['label'].tolist()
groups = df['slide'].tolist()

# Hyperparameters
batch_size = 8
num_epochs = 100
early_stopping_patience = 15
criterion = nn.BCEWithLogitsLoss()
strat_group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

# Cross-validation
for fold, (train_idx, val_idx) in enumerate(strat_group_kfold.split(images, labels, groups)):
    logger.info(f"Starting Fold {fold + 1}")

    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = EmbeddingDataset(train_images, train_labels)
    val_dataset = EmbeddingDataset(val_images, val_labels)

    class_counts = [train_labels.count(i) for i in range(2)]
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = HoptimusBinaryClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # We want to maximize validation accuracy
        factor=0.5,           # Reduce LR by half
        patience=3,           # Wait 3 epochs before reducing
        min_lr=1e-7,          # Don't go below this LR
        verbose=True          # Print when LR changes
    )

    best_val_bal_acc = 0.0
    early_stop_counter = 0
    best_model_path = f'hoptimus0_linear_probe_fold_{fold + 1}_best.pth'

    for epoch in range(num_epochs):
        model.train()
        train_loss, preds, targets = 0.0, [], []

        for x, y in tqdm(train_loader, desc=f"Fold {fold + 1} - Epoch {epoch + 1} Training"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            targets.extend(y.cpu().numpy())

        train_bal_acc = balanced_accuracy_score(targets, preds)

        # Validation
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Fold {fold + 1} - Epoch {epoch + 1} Validation"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_bal_acc = balanced_accuracy_score(val_targets, val_preds)

        logger.info(
            f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Bal Acc: {train_bal_acc:.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            f"Val Bal Acc: {val_bal_acc:.4f}"
        )

        scheduler.step(val_bal_acc)

        # Early stopping
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save(model, best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    logger.info(f"Fold {fold + 1} - Best Val Balanced Accuracy: {best_val_bal_acc:.4f}")
    fold_accuracies.append(best_val_bal_acc)

# Final results
avg_bal_acc = sum(fold_accuracies) / len(fold_accuracies)
logger.info(f"Average Validation Balanced Accuracy across all folds: {avg_bal_acc:.4f}")
