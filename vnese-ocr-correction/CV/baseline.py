#!/usr/bin/env parser
import os
import argparse
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Headless mode for matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# ==========================================
# 1. Dataset Definition
# ==========================================
class NomOCRDataset(Dataset):
    """
    Dataset class for Nom OCR single-character classification.
    """
    def __init__(self, root_dir, dataframe, char2idx=None, transform=None, is_train=True):
        self.root_dir = root_dir
        self.dataframe = dataframe.reset_index(drop=True)
        self.char2idx = char2idx
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row['image']
        
        if self.is_train:
            img_path = os.path.join(self.root_dir, "train", "images", img_name)
            label_char = row['label']
            label_idx = self.char2idx[label_char]
        else:
            img_path = os.path.join(self.root_dir, "public_test", "images", img_name)
            label_idx = -1  # Not used during test inference
            
        # Convert to RGB to make it compatible with default 3-channel torchvision models
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

# ==========================================
# 2. Visualization Functions
# ==========================================
def save_sample_grid(dataset, idx2char, output_path="train_samples.png"):
    """
    Saves a 4x4 grid of sample training images with their labels.
    """
    plt.figure(figsize=(10, 10))
    indices = random.sample(range(len(dataset)), min(16, len(dataset)))
    for grid_idx, dataset_idx in enumerate(indices):
        img_tensor, label_idx = dataset[dataset_idx]
        # Unnormalize image tensor to display it [0, 1]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        img = np.clip(img, 0, 1)
        
        plt.subplot(4, 4, grid_idx + 1)
        plt.imshow(img)
        char_label = idx2char[label_idx]
        
        # Catch font issues gracefully (matplotlib default might render tofus for CJK/Nom characters)
        try:
            plt.title(f"Label: {char_label}")
        except Exception:
            plt.title("Label: [CJK]")
            
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[*] Visualized training samples saved to: {output_path}")

def save_metrics_plot(train_losses, val_losses, train_f1s, val_f1s, output_path="metrics.png"):
    """
    Plots and saves loss and macro F1 curves over training epochs.
    """
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curves')
    plt.legend()
    plt.grid(True)
    
    # Macro F1 Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_f1s, label='Train Macro F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val Macro F1', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.title('Macro F1 curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[*] Training metrics plot saved to: {output_path}")

# ==========================================
# 3. Model Loader
# ==========================================
def get_model(num_classes, pretrained=True):
    """
    Instantiates ResNet-18 and adapts the head for Nom classification.
    """
    if pretrained:
        if hasattr(models, 'ResNet18_Weights'):
            # Modern torchvision API
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(pretrained=True)
        print("[*] Loaded pre-trained ResNet-18 weights.")
    else:
        if hasattr(models, 'ResNet18_Weights'):
            model = models.resnet18(weights=None)
        else:
            model = models.resnet18(pretrained=False)
        print("[*] Initialized ResNet-18 from scratch.")
        
    # Replace final classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ==========================================
# 4. Main Training and Submission Script
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Baseline for Nom Character OCR Challenge")
    parser.add_argument("--mode", type=str, default="both", choices=["train", "submit", "both"],
                        help="Mode: 'train' to train model, 'submit' to generate submission, 'both' to do both.")
    parser.add_argument("--data-dir", type=str, default="dataset",
                        help="Path to the challenge dataset directory containing train/ and public_test/")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=64, help="Height and Width to resize character images")
    parser.add_argument("--pretrained", type=str, default="true", choices=["true", "false"],
                        help="Use pre-trained ImageNet weights (recommended for transfer learning)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Path to save output files (vocab.json, best_model.pth, metrics.png, train_samples.png, submission.csv)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Establish save paths
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    model_weight_path = os.path.join(args.output_dir, "best_model.pth")
    samples_plot_path = os.path.join(args.output_dir, "train_samples.png")
    metrics_plot_path = os.path.join(args.output_dir, "metrics.png")
    submission_path = os.path.join(args.output_dir, "submission.csv")
    
    use_pretrained = (args.pretrained.lower() == "true")
    
    # Define image transformations
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomRotation(10),  # Slight rotation for data augmentation
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] range
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # ==========================================
    # Train / Both Mode
    # ==========================================
    if args.mode in ["train", "both"]:
        labels_file = os.path.join(args.data_dir, "train", "labels.csv")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Missing train labels at: {labels_file}")
            
        print("[*] Reading training data...")
        df = pd.read_csv(labels_file)
        
        # Build character vocabulary dynamically
        unique_chars = sorted(list(df['label'].unique()))
        num_classes = len(unique_chars)
        char2idx = {char: idx for idx, char in enumerate(unique_chars)}
        idx2char = {idx: char for idx, char in enumerate(unique_chars)}
        
        # Save vocabulary for inference consistency
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": unique_chars}, f, ensure_ascii=False, indent=2)
        print(f"[*] Vocabulary built with {num_classes} unique characters. Saved to: {vocab_path}")
        
        # Perform robust Stratified train-test split
        counts = df['label'].value_counts()
        single_samples = counts[counts == 1].index
        
        df_single = df[df['label'].isin(single_samples)]
        df_multi = df[~df['label'].isin(single_samples)]
        
        if len(df_multi) > 0:
            train_multi, val_multi = train_test_split(
                df_multi, test_size=0.25, random_state=42, stratify=df_multi['label']
            )
            train_df = pd.concat([train_multi, df_single]).reset_index(drop=True)
            val_df = val_multi.reset_index(drop=True)
        else:
            train_df = df_single.reset_index(drop=True)
            val_df = pd.DataFrame(columns=df.columns)
            
        print(f"[*] Dataset split: {len(train_df)} training samples, {len(val_df)} validation samples.")
        
        train_dataset = NomOCRDataset(args.data_dir, train_df, char2idx, transform=train_transform, is_train=True)
        val_dataset = NomOCRDataset(args.data_dir, val_df, char2idx, transform=val_transform, is_train=True)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Visualize training sample grid
        save_sample_grid(train_dataset, idx2char, samples_plot_path)
        
        # Load model, criterion, optimizer, scheduler
        model = get_model(num_classes, pretrained=use_pretrained).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Metric tracking lists
        train_losses, val_losses = [], []
        train_f1s, val_f1s = [], []
        best_val_f1 = -1.0
        
        print(f"[*] Starting training for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            # --- TRAINING PHASE ---
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
            for images, targets in train_pbar:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                train_preds.extend(predicted.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
                train_total += targets.size(0)
                
                # Approximate running accuracy for progress bar
                running_acc = np.mean(np.array(train_preds) == np.array(train_targets))
                train_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * running_acc:.2f}%"
                })
                
            epoch_train_loss = train_loss / train_total
            epoch_train_f1 = f1_score(train_targets, train_preds, average='macro', zero_division=0)
            
            # --- VALIDATION PHASE ---
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            val_total = 0
            
            if len(val_loader) > 0:
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]")
                with torch.no_grad():
                    for images, targets in val_pbar:
                        images, targets = images.to(device), targets.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item() * images.size(0)
                        _, predicted = outputs.max(1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_targets.extend(targets.cpu().numpy())
                        val_total += targets.size(0)
                        
                        running_acc = np.mean(np.array(val_preds) == np.array(val_targets))
                        val_pbar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{100.0 * running_acc:.2f}%"
                        })
                epoch_val_loss = val_loss / val_total
                epoch_val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
            else:
                epoch_val_loss = 0.0
                epoch_val_f1 = 0.0
                
            scheduler.step()
            
            # Log metrics
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_f1s.append(epoch_train_f1)
            val_f1s.append(epoch_val_f1)
            
            print(f"Summary Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.4f} | Train Macro F1: {epoch_train_f1:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Macro F1: {epoch_val_f1:.4f}")
            
            # Save checkpoint if F1 score improves
            if epoch_val_f1 > best_val_f1:
                best_val_f1 = epoch_val_f1
                torch.save(model.state_dict(), model_weight_path)
                print(f"[*] Saved new best model checkpoint to: {model_weight_path} (F1: {best_val_f1:.4f})")
                
        # Generate metrics plot
        save_metrics_plot(train_losses, val_losses, train_f1s, val_f1s, metrics_plot_path)
        print("[*] Training completed.")
        
    # ==========================================
    # Submit / Both Mode
    # ==========================================
    if args.mode in ["submit", "both"]:
        print("[*] Starting test submission generation...")
        
        # Load vocab.json
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Missing vocab.json at: {vocab_path}. Run training mode first to generate vocab.")
            
        with open(vocab_path, "r", encoding="utf-8") as f:
            unique_chars = json.load(f)["vocab"]
            
        num_classes = len(unique_chars)
        idx2char = {idx: char for idx, char in enumerate(unique_chars)}
        
        # Instantiate and load model
        model = get_model(num_classes, pretrained=False).to(device)
        if not os.path.exists(model_weight_path):
            raise FileNotFoundError(f"Missing weights file at: {model_weight_path}. Run training mode first to save checkpoint.")
            
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print(f"[*] Loaded best model weights from: {model_weight_path}")
        
        # Read sample submission
        sample_sub_path = os.path.join(args.data_dir, "public_test", "sample_submission.csv")
        if not os.path.exists(sample_sub_path):
            raise FileNotFoundError(f"Missing sample submission file at: {sample_sub_path}")
            
        submit_df = pd.read_csv(sample_sub_path)
        print(f"[*] Loaded sample submission containing {len(submit_df)} entries.")
        
        test_dataset = NomOCRDataset(args.data_dir, submit_df, transform=val_transform, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        model.eval()
        predictions = []
        
        test_pbar = tqdm(test_loader, desc="Generating submission")
        with torch.no_grad():
            for images, _ in test_pbar:
                images = images.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                predictions.extend(preds.cpu().numpy())
                
        # Map predictions to character labels
        predicted_chars = [idx2char[idx] for idx in predictions]
        submit_df['label'] = predicted_chars
        
        # Save submission
        submit_df.to_csv(submission_path, index=False)
        print(f"[*] Successfully saved final submission file to: {submission_path}")

if __name__ == "__main__":
    main()
