import os
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from utils import set_seed
from dataset import SignDataset
from model import CNNLSTMAttn

DATA_ROOT = "data"
TRAIN_DIR = f"{DATA_ROOT}/train"
LABEL_PATH = f"{DATA_ROOT}/label_mapping.pkl"

CKPT_PATH = "outputs/best.pth"

def load_label_map(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def evaluate(model, loader, device, criterion):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            with autocast("cuda"):
                logits = model(frames)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average="macro")
    return total_loss / len(loader), f1

def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("GPUs:", torch.cuda.device_count())

    label_map = load_label_map(LABEL_PATH)
    num_classes = len(label_map)

    dataset = SignDataset(TRAIN_DIR, label_map, max_frames=48)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=4, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = CNNLSTMAttn(num_classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=1)

    scaler = GradScaler()

    best_f1 = 0
    patience = 4
    bad_epochs = 0

    for epoch in range(36):
        model.train()
        total_loss = 0

        for frames, labels in train_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model(frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        val_loss, f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch} | Loss {total_loss:.4f} | F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1

            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), CKPT_PATH)
            else:
                torch.save(model.state_dict(), CKPT_PATH)

            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping")
            break

    print("Best F1:", best_f1)
    
if __name__ == "__main__":
    main()