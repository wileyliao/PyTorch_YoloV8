from __future__ import annotations
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# 固定參數
TRAIN_DIR   = Path(r"C:\database\tzuchi\20250905\output\20250905\train_test_split_gray_he_244\train")
VAL_DIR     = Path(r"C:\database\tzuchi\20250905\output\20250905\train_test_split_gray_he_244\val")
CKPT_DIR    = Path("checkpoints")

IMG_SIZE    = 244
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1.5e-3
WEIGHT_DECAY= 0.01
PATIENCE    = 10
NUM_WORKERS = 0
USE_AMP     = True

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, mode="min", delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = None
        self.counter = 0
        self.stop = False
    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return
        improved = (metric < self.best - self.delta) if self.mode == "min" else (metric > self.best + self.delta)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# DataLoader
def get_loaders():
    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(360),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tfms)
    val_ds   = datasets.ImageFolder(str(VAL_DIR),   transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

# Model
def build_model(num_classes: int) -> nn.Module:
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1  # torchvision>=0.13
    except Exception:
        weights = "IMAGENET1K_V1"
    model = models.efficientnet_b0(weights=weights)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    return model

# Train / Validate
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if USE_AMP:
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

# Main
def main():
    from efficientNet_utils import write_class_json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = get_loaders()
    print(f"📚 類別數：{len(classes)} -> {classes}")

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)
    early = EarlyStopping(patience=PATIENCE, mode="min", delta=0.0)
    best_val_loss = float("inf")

    ckpt_best = CKPT_DIR / f"efficientnet_b0_best_{IMG_SIZE}.pth"
    ckpt_last = CKPT_DIR / f"efficientnet_b0_last_{IMG_SIZE}.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_best)
        print(f"[{epoch:03d}/{EPOCHS}] "
              f"Train Loss {train_loss:.4f} | Acc {train_acc*100:.2f}%  ||  "
              f"Val Loss {val_loss:.4f} | Acc {val_acc*100:.2f}%  "
              f"{'**BEST**' if is_best else ''}")
        early(val_loss)
        if early.stop:
            print(f"\n⏹️ Early stopping triggered (patience={PATIENCE}). 最佳 Val Loss: {best_val_loss:.4f}")
            break

    torch.save(model.state_dict(), ckpt_last)
    print("\n✅ 完成：best/last 權重已儲存到", CKPT_DIR)

    # 輸出 JSON：依 ckpt_best 命名
    if ckpt_best.exists():
        write_class_json(TRAIN_DIR, ckpt_best)
    else:
        write_class_json(TRAIN_DIR, ckpt_last)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
