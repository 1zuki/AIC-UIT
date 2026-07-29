"""
Nôm Character Recognition.

Single-GPU image classification pipeline for the challenge format:

    dataset/
    ├── train/
    │   ├── images/
    │   └── labels.csv      # image,label
    └── test/
        └── images/

Output:
    submission.zip
    └── submission.csv      # image,label

defaults:
  - uses only training labels/images for training/validation
  - no test pseudo-labeling
  - no external data
  - torchvision ImageNet weights are supported because the statement allows them
  - single GPU, checkpointing, resume, AMP, EMA, TTA inference
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF

try:
    from PIL import Image, ImageOps
except Exception as exc:
    raise RuntimeError("Pillow/PIL is required by torchvision image loading.") from exc


# utils
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_csv_rows(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_submission_csv(path: Path, rows: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "label"])
        for image, label in rows:
            w.writerow([image, label])


def find_data_dir(root: Path) -> Path:
    """ accept either the dataset dir itself or a parent containing CV/dataset """
    root = Path(root)
    candidates = [
        root,
        root / "dataset",
        root / "CV" / "dataset",
    ]
    for p in candidates:
        if (p / "train" / "labels.csv").exists() and (p / "train" / "images").is_dir():
            return p
    # if files are extracted with nested folders, scan a little
    for p in root.rglob("labels.csv") if root.exists() else []:
        if p.parent.name == "train" and (p.parent / "images").is_dir():
            return p.parent.parent

    raise FileNotFoundError(
        f"Could not find dataset under {root}. Need train/labels.csv and train/images/."
    )


def find_test_images_dir(data_dir: Path) -> Path:
    for rel in ["test/images", "public_test/images", "private_test/images"]:
        p = data_dir / rel
        if p.is_dir():
            return p

    raise FileNotFoundError(f"Could not find test images under {data_dir}/test/images")


def list_images(folder: Path) -> List[str]:
    return sorted([p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# transform image
class SquarePad:
    """ pad a PIL image to square, preserving aspect ratio before resize """
    def __init__(self, fill: int = 255):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img

        side = max(w, h)
        left = (side - w) // 2
        top = (side - h) // 2
        right = side - w - left
        bottom = side - h - top
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=self.fill)


class RandomInvert:
    def __init__(self, p: float = 0.0):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() >= self.p:
            return img
        
        return ImageOps.invert(img.convert("RGB"))


def build_transforms(img_size: int, train: bool, pretrained: bool, invert_aug_prob: float = 0.03):
    mean = (0.485, 0.456, 0.406) if pretrained else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if pretrained else (0.5, 0.5, 0.5)
    if train:
        return transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB")),
            SquarePad(fill=255),
            transforms.RandomResizedCrop(img_size, scale=(0.72, 1.0), ratio=(0.88, 1.12), antialias=True),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.08, 0.08),
                    scale=(0.88, 1.12),
                    shear=(-4, 4, -4, 4),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=255,
                )
            ], p=0.85),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.12),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.20, contrast=0.25)], p=0.55),
            RandomInvert(p=invert_aug_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        SquarePad(fill=255),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def preprocess_pil(img: Image.Image, img_size: int, pretrained: bool) -> torch.Tensor:
    return build_transforms(img_size, train=False, pretrained=pretrained)(img)


def make_tta_views(img: Image.Image, img_size: int, pretrained: bool, tta: int) -> torch.Tensor:
    """Deterministic no-flip TTA; flips are unsafe for characters."""
    img = img.convert("RGB")
    views: List[Image.Image] = [img]
    if tta >= 3:
        views += [
            TF.affine(img, angle=0, translate=[2, 0], scale=1.00, shear=[0, 0], fill=255),
            TF.affine(img, angle=0, translate=[-2, 0], scale=1.00, shear=[0, 0], fill=255),
        ]
    if tta >= 5:
        views += [
            TF.affine(img, angle=0, translate=[0, 2], scale=1.00, shear=[0, 0], fill=255),
            TF.affine(img, angle=0, translate=[0, -2], scale=1.00, shear=[0, 0], fill=255),
        ]
    if tta >= 9:
        views += [
            TF.affine(img, angle=2, translate=[0, 0], scale=1.02, shear=[0, 0], fill=255),
            TF.affine(img, angle=-2, translate=[0, 0], scale=1.02, shear=[0, 0], fill=255),
            TF.affine(img, angle=0, translate=[0, 0], scale=0.96, shear=[0, 0], fill=255),
            TF.affine(img, angle=0, translate=[0, 0], scale=1.06, shear=[0, 0], fill=255),
        ]
    tensors = [preprocess_pil(v, img_size, pretrained) for v in views[:max(1, tta)]]
    return torch.stack(tensors, dim=0)


# dataset
class NomDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        rows: Sequence[dict],
        char2idx: Optional[Dict[str, int]],
        transform,
        train: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.rows = list(rows)
        self.char2idx = char2idx
        self.transform = transform
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        name = row["image"]
        path = self.image_dir / name
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        if self.train:
            y = self.char2idx[row["label"]]
            return x, torch.tensor(y, dtype=torch.long), name
        return x, torch.tensor(-1, dtype=torch.long), name


class TestTTADataset(Dataset):
    def __init__(self, image_dir: Path, image_names: Sequence[str], img_size: int, pretrained: bool, tta: int):
        self.image_dir = Path(image_dir)
        self.image_names = list(image_names)
        self.img_size = int(img_size)
        self.pretrained = bool(pretrained)
        self.tta = int(tta)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        img = Image.open(self.image_dir / name).convert("RGB")
        x = make_tta_views(img, self.img_size, self.pretrained, self.tta)
        return x, name


def stratified_split(rows: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    train, val = [], []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n < 2 or val_ratio <= 0:
            train.extend(items)
            continue
        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_vocab(rows: List[dict]) -> Tuple[Dict[str, int], List[str]]:
    labels = sorted({r["label"] for r in rows})
    return {c: i for i, c in enumerate(labels)}, labels


def label_counts(rows: Sequence[dict], char2idx: Dict[str, int]) -> torch.Tensor:
    counts = torch.zeros(len(char2idx), dtype=torch.float32)
    for r in rows:
        counts[char2idx[r["label"]]] += 1
    return counts


# model
def _get_weights_enum(name: str):
    mapping = {
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "efficientnet_b0": "EfficientNet_B0_Weights",
        "efficientnet_b2": "EfficientNet_B2_Weights",
        "efficientnet_v2_s": "EfficientNet_V2_S_Weights",
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "swin_t": "Swin_T_Weights",
    }
    enum_name = mapping.get(name)
    if enum_name and hasattr(models, enum_name):
        return getattr(models, enum_name).DEFAULT
    return None


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.15) -> nn.Module:
    name = name.lower()
    weights = _get_weights_enum(name) if pretrained else None
    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        
        if dropout is not None:
            model.classifier[0].p = dropout

    elif name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        model.classifier[0].p = dropout
    elif name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        model.classifier[0].p = dropout
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        # classifier usually LayerNorm2d, Flatten, Linear; no dropout by default
    elif name == "swin_t":
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model '{name}'. Try convnext_tiny/resnet50/efficientnet_b0/efficientnet_v2_s/swin_t")
    return model


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.non_float = {k: v.detach().clone() for k, v in model.state_dict().items() if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            elif not v.dtype.is_floating_point:
                self.non_float[k] = v.detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out = {k: v.detach().clone() for k, v in self.shadow.items()}
        out.update({k: v.detach().clone() for k, v in self.non_float.items()})
        return out

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone() for k, v in state.items() if v.dtype.is_floating_point}
        self.non_float = {k: v.detach().clone() for k, v in state.items() if not v.dtype.is_floating_point}


# metrics
class FocalCE(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 1.0, label_smoothing: float = 0.03):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.gamma <= 0:
            return ce.mean()

        pt = torch.exp(-ce.detach())
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def macro_f1_score(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    if len(y_true) == 0:
        return 0.0

    true = torch.tensor(y_true, dtype=torch.long)
    pred = torch.tensor(y_pred, dtype=torch.long)
    present = torch.unique(true)
    f1s = []
    for c in present.tolist():
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)

    return float(sum(f1s) / max(1, len(f1s)))


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    return float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true))


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def state_dict(self):
        return {"step_num": self.step_num, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.step_num = int(state.get("step_num", 0))
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))
        self._apply_lr()

    def _ratio(self):
        if self.step_num < self.warmup_steps:
            return max(1e-8, self.step_num / self.warmup_steps)

        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply_lr(self):
        ratio = self._ratio()
        for g, lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = lr * ratio

    def step(self):
        self.step_num += 1
        self._apply_lr()


"""
train
eval
infer
"""
@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    model: str = "convnext_tiny"
    img_size: int = 224
    pretrained: bool = True
    epochs: int = 40
    batch_size: int = 32
    accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.05
    val_ratio: float = 0.18
    seed: int = 42
    num_workers: int = 4
    label_smoothing: float = 0.04
    focal_gamma: float = 0.7
    dropout: float = 0.15
    balanced_sampler: bool = True
    amp: bool = True
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    save_every: int = 5
    invert_aug_prob: float = 0.02


def make_loaders(cfg: TrainConfig, char2idx: Dict[str, int], train_rows: List[dict], val_rows: List[dict]):
    data_dir = Path(cfg.data_dir)
    train_img_dir = data_dir / "train" / "images"
    train_tf = build_transforms(cfg.img_size, train=True, pretrained=cfg.pretrained, invert_aug_prob=cfg.invert_aug_prob)
    val_tf = build_transforms(cfg.img_size, train=False, pretrained=cfg.pretrained)
    train_ds = NomDataset(train_img_dir, train_rows, char2idx, train_tf, train=True)
    val_ds = NomDataset(train_img_dir, val_rows, char2idx, val_tf, train=True)

    if cfg.balanced_sampler:
        counts = label_counts(train_rows, char2idx)
        sample_weights = []
        for r in train_rows:
            c = counts[char2idx[r["label"]]].item()
            sample_weights.append(1.0 / max(1.0, math.sqrt(c)))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int, criterion=None) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    total = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)

        pred = logits.argmax(1)
        preds.extend(pred.detach().cpu().tolist())
        targets.extend(y.detach().cpu().tolist())
        total += x.size(0)
    return {
        "loss": total_loss / max(1, total),
        "acc": accuracy_score(targets, preds),
        "macro_f1": macro_f1_score(targets, preds, num_classes),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: Optional[ModelEMA],
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_f1: float,
    cfg: TrainConfig,
    labels: List[str],
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_f1": best_f1,
            "config": asdict(cfg),
            "labels": labels,
            "metrics": metrics,
        },
        path,
    )


def append_log(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def train_main(args) -> None:
    seed_everything(args.seed)
    data_dir = find_data_dir(Path(args.data_dir))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(data_dir / "train" / "labels.csv")
    if not rows or "image" not in rows[0] or "label" not in rows[0]:
        raise ValueError("labels.csv must have columns: image,label")

    # Check missing files early.
    image_dir = data_dir / "train" / "images"
    missing = [r["image"] for r in rows[:2000] if not (image_dir / r["image"]).exists()]
    if missing:
        raise FileNotFoundError(f"Some training images are missing, e.g. {missing[:5]}")

    char2idx, labels = build_vocab(rows)
    train_rows, val_rows = stratified_split(rows, args.val_ratio, args.seed)
    cfg = TrainConfig(
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        model=args.model,
        img_size=args.img_size,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        dropout=args.dropout,
        balanced_sampler=not args.no_balanced_sampler,
        amp=not args.no_amp,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        save_every=args.save_every,
        invert_aug_prob=args.invert_aug_prob,
    )
    save_json(out_dir / "label_vocab.json", {"labels": labels, "char2idx": char2idx})
    save_json(out_dir / "config.json", asdict(cfg))

    print(f"DATA_DIR={data_dir}")
    print(f"OUTPUT_DIR={out_dir}")
    print(f"rows={len(rows)} train={len(train_rows)} val={len(val_rows)} classes={len(labels)}")
    print(f"model={cfg.model} img_size={cfg.img_size} pretrained={cfg.pretrained}")

    train_loader, val_loader = make_loaders(cfg, char2idx, train_rows, val_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = build_model(cfg.model, len(labels), pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)
    counts = label_counts(train_rows, char2idx)
    class_weights = (1.0 / torch.sqrt(torch.clamp(counts, min=1.0)))
    class_weights = class_weights / class_weights.mean()
    criterion = FocalCE(class_weights.to(device), gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / cfg.accum_steps) * cfg.epochs)
    scheduler = WarmupCosine(optimizer, warmup_steps=max(20, int(0.06 * total_steps)), total_steps=total_steps)
    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

    start_epoch = 1
    best_f1 = -1.0
    if args.resume:
        ckpt = torch_load(Path(args.resume), map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        if ema is not None and ckpt.get("ema_state") is not None:
            ema.load_state_dict(ckpt["ema_state"])
        if ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_f1={best_f1:.5f}")

    log_path = out_dir / "train_log.csv"

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        train_preds, train_targets = [], []
        step_in_epoch = 0

        for batch_idx, (x, y, _) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=(cfg.amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y) / cfg.accum_steps

            scaler.scale(loss).backward()

            if (batch_idx % cfg.accum_steps == 0) or (batch_idx == len(train_loader)):
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update(model)

                step_in_epoch += 1

            with torch.no_grad():
                pred = logits.argmax(1)
                train_preds.extend(pred.detach().cpu().tolist())
                train_targets.extend(y.detach().cpu().tolist())
                running_loss += float(loss.item()) * cfg.accum_steps * x.size(0)
                seen += x.size(0)

            if batch_idx == 1 or batch_idx % max(1, len(train_loader) // 10) == 0:
                acc = accuracy_score(train_targets, train_preds)
                print(
                    f"epoch {epoch:03d}/{cfg.epochs:03d} batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"loss={running_loss/max(1,seen):.4f} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        train_metrics = {
            "loss": running_loss / max(1, seen),
            "acc": accuracy_score(train_targets, train_preds),
            "macro_f1": macro_f1_score(train_targets, train_preds, len(labels)),
        }

        eval_model = model
        raw_state = None

        if ema is not None:
            raw_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict(), strict=True)
            eval_model = model

        val_metrics = evaluate(eval_model, val_loader, device, len(labels), criterion=criterion)
        if raw_state is not None:
            model.load_state_dict(raw_state, strict=True)

        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_acc": f"{train_metrics['acc']:.6f}",
            "train_macro_f1": f"{train_metrics['macro_f1']:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_acc": f"{val_metrics['acc']:.6f}",
            "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.8g}",
            "seconds": f"{elapsed:.1f}",
        }
        append_log(log_path, row)
        print(
            f"EPOCH {epoch:03d} | train_f1={train_metrics['macro_f1']:.5f} "
            f"val_f1={val_metrics['macro_f1']:.5f} val_acc={val_metrics['acc']*100:.2f}% "
            f"best={best_f1:.5f} time={elapsed/60:.1f}m",
            flush=True,
        )

        metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(out_dir / "last.pt", model, ema, optimizer, scheduler, scaler, epoch, best_f1, cfg, labels, metrics)
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", model, ema, optimizer, scheduler, scaler, epoch, best_f1, cfg, labels, metrics)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(out_dir / "best.pt", model, ema, optimizer, scheduler, scaler, epoch, best_f1, cfg, labels, metrics)
            print(f"Saved new best.pt with val_macro_f1={best_f1:.5f}")

    print("Training finished. Best checkpoint:", out_dir / "best.pt")


def load_checkpoint_model(ckpt_path: Path, device: torch.device):
    ckpt = torch_load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("config", {})
    labels = ckpt["labels"]
    model_name = cfg_dict.get("model", "convnext_tiny")
    pretrained = bool(cfg_dict.get("pretrained", True))
    dropout = float(cfg_dict.get("dropout", 0.15))
    img_size = int(cfg_dict.get("img_size", 224))
    model = build_model(model_name, len(labels), pretrained=False, dropout=dropout).to(device)
    state = ckpt.get("ema_state") if ckpt.get("ema_state") is not None else ckpt["model_state"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, labels, img_size, pretrained, cfg_dict


@torch.no_grad()
def infer_main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device)
    data_dir = find_data_dir(Path(args.data_dir))
    test_dir = find_test_images_dir(data_dir)

    ckpt_paths = [Path(p) for p in args.checkpoints]
    models_loaded = []
    labels_ref = None
    img_size_ref = None
    pretrained_ref = None

    for p in ckpt_paths:
        model, labels, img_size, pretrained, cfg_dict = load_checkpoint_model(p, device)
        print(f"Loaded {p} | model={cfg_dict.get('model')} epoch={torch_load(p, map_location='cpu').get('epoch')} labels={len(labels)}")
        if labels_ref is None:
            labels_ref = labels
            img_size_ref = img_size
            pretrained_ref = pretrained
        elif labels != labels_ref:
            raise ValueError("All checkpoints must have the same label vocabulary/order.")

        models_loaded.append(model)

    labels = labels_ref
    idx2char = {i: c for i, c in enumerate(labels)}

    # preserve sample_submission/test order if available; otherwise sort image names
    image_names = None
    for sample_rel in ["test/sample_submission.csv", "public_test/sample_submission.csv", "sample_submission.csv"]:
        sample_path = data_dir / sample_rel
        if sample_path.exists():
            sample_rows = read_csv_rows(sample_path)
            if sample_rows and "image" in sample_rows[0]:
                image_names = [r["image"] for r in sample_rows]
                print("Using image order from", sample_path)
                break

    if image_names is None:
        image_names = list_images(test_dir)
        print("Using sorted image order from", test_dir)

    ds = TestTTADataset(test_dir, image_names, img_size_ref, pretrained_ref, tta=args.tta)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows_out: List[Tuple[str, str]] = []
    for batch_idx, (x, names) in enumerate(loader, start=1):
        # x: B,TTA,C,H,W -> flatten TTA, average logits per image
        b, t = x.shape[:2]
        x = x.view(b * t, *x.shape[2:]).to(device, non_blocking=True)
        logits_sum = None

        for model in models_loaded:
            logits = model(x).view(b, t, -1).mean(dim=1)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        pred = logits_sum.argmax(1).detach().cpu().tolist()
        for name, idx in zip(names, pred):
            rows_out.append((str(name), idx2char[int(idx)]))
        if batch_idx == 1 or batch_idx % max(1, len(loader) // 10) == 0:
            print(f"infer batch {batch_idx}/{len(loader)}")

    sub_path = Path(args.submission_csv)
    write_submission_csv(sub_path, rows_out)
    print(f"Wrote {sub_path} with {len(rows_out)} rows")

    if args.zip:
        zip_path = Path(args.zip)
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(sub_path, arcname="submission.csv")
        print(f"Wrote {zip_path} ({zip_path.stat().st_size/1024:.1f} KB), containing only submission.csv")


def zip_main(args) -> None:
    sub_path = Path(args.submission_csv)
    zip_path = Path(args.zip)

    if not sub_path.exists():
        raise FileNotFoundError(sub_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(sub_path, arcname="submission.csv")
    print(f"Wrote {zip_path} ({zip_path.stat().st_size/1024:.1f} KB), containing only submission.csv")


def inspect_main(args) -> None:
    data_dir = find_data_dir(Path(args.data_dir))
    rows = read_csv_rows(data_dir / "train" / "labels.csv")
    labels = sorted({r["label"] for r in rows})
    counts = defaultdict(int)

    for r in rows:
        counts[r["label"]] += 1

    test_dir = find_test_images_dir(data_dir)
    print("DATA_DIR:", data_dir)
    print("train images:", len(rows))
    print("test images:", len(list_images(test_dir)))
    print("classes:", len(labels))
    vals = sorted(counts.values())
    if vals:
        print("class count min/median/max:", vals[0], vals[len(vals)//2], vals[-1])
        rare = sum(1 for v in vals if v == 1)
        print("single-sample classes:", rare)
    print("first labels:", labels[:20])


# cli
def parse_args():
    p = argparse.ArgumentParser(description="Nôm character recognition cooker")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect")
    pi.add_argument("--data-dir", default="CV/dataset")

    pt = sub.add_parser("train")
    pt.add_argument("--data-dir", default="CV/dataset")
    pt.add_argument("--output-dir", default="runs/nom_convnext")
    pt.add_argument("--model", default="convnext_tiny", choices=["convnext_tiny", "resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b2", "efficientnet_v2_s", "swin_t"])
    pt.add_argument("--img-size", type=int, default=224)
    pt.add_argument("--no-pretrained", action="store_true")
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--accum-steps", type=int, default=1)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight-decay", type=float, default=0.05)
    pt.add_argument("--val-ratio", type=float, default=0.18)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--num-workers", type=int, default=4)
    pt.add_argument("--label-smoothing", type=float, default=0.04)
    pt.add_argument("--focal-gamma", type=float, default=0.7)
    pt.add_argument("--dropout", type=float, default=0.15)
    pt.add_argument("--no-balanced-sampler", action="store_true")
    pt.add_argument("--no-amp", action="store_true")
    pt.add_argument("--ema-decay", type=float, default=0.999)
    pt.add_argument("--grad-clip", type=float, default=1.0)
    pt.add_argument("--save-every", type=int, default=5)
    pt.add_argument("--invert-aug-prob", type=float, default=0.02)
    pt.add_argument("--resume", default=None)

    pf = sub.add_parser("infer")
    pf.add_argument("--data-dir", default="CV/dataset")
    pf.add_argument("--checkpoints", nargs="+", required=True)
    pf.add_argument("--submission-csv", default="submission.csv")
    pf.add_argument("--zip", default="submission.zip")
    pf.add_argument("--batch-size", type=int, default=64)
    pf.add_argument("--num-workers", type=int, default=4)
    pf.add_argument("--tta", type=int, default=5, help="1, 3, 5, or 9. No flips are used.")
    pf.add_argument("--cpu", action="store_true")

    pz = sub.add_parser("zip")
    pz.add_argument("--submission-csv", default="submission.csv")
    pz.add_argument("--zip", default="submission.zip")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "inspect":
        inspect_main(args)
    elif args.cmd == "train":
        train_main(args)
    elif args.cmd == "infer":
        infer_main(args)
    elif args.cmd == "zip":
        zip_main(args)
    else:  # pragma: no cover
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
