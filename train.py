"""
Training script for the Music Genre Classifier.

Usage:
    python train.py                   # train for 30 epochs (default)
    python train.py --epochs 5        # quick smoke test
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import GenreDataset, get_file_list, prepare_dataset
from model import GenreCNN


# ─────────────────── Helpers ────────────────────────────────────


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="  train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="  val  ", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ─────────────────── Main ──────────────────────────────────────


def main(epochs: int | None = None):
    epochs = epochs or config.EPOCHS
    device = get_device()
    print(f"[⚙] Device: {device}")

    # 1. Prepare data
    prepare_dataset()
    file_list = get_file_list()

    # 2. Stratified train/val split (by file, not segment)
    labels = [lbl for _, lbl in file_list]
    train_files, val_files = train_test_split(
        file_list,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )
    print(f"[i] Train files: {len(train_files)} | Val files: {len(val_files)}")

    train_ds = GenreDataset(train_files)
    val_ds = GenreDataset(val_files)
    print(f"[i] Train segments: {len(train_ds)} | Val segments: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # 3. Model, loss, optimizer
    model = GenreCNN(
        num_classes=config.NUM_CLASSES, dropout=config.DROPOUT
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )

    # 4. Training loop
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f" Training for {epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            tag = " ★ best"

        print(
            f"Epoch {epoch:3d}/{epochs}  │  "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f}  │  "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}  │  "
            f"{elapsed:.1f}s{tag}"
        )

    print(f"\n{'='*60}")
    print(f" Best validation accuracy: {best_val_acc:.4f}")
    print(f" Model saved to {config.BEST_MODEL_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Music Genre Classifier")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (default: from config)")
    args = parser.parse_args()
    main(epochs=args.epochs)
