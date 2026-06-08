#!/usr/bin/env python3
"""Baseline training driver.

Builds DataLoaders from the cached HaGRIDv2 dataset, trains the model, and saves
a .pth checkpoint — the fixed handoff format consumed by src/compression:

    {model_state_dict, model_cfg, label_map, val_acc}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import HaGRIDv2Dataset
from src.models.test import build_model   # baseline model; swap for real one later

LABEL_MAP = {0: "N/A", 1: "fist", 2: "like", 3: "ok", 4: "one", 5: "palm"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann_root", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--cache_root", default="data/processed")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--crop_size", type=int, default=112)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_loaders(args):
    train_ds = HaGRIDv2Dataset(args.ann_root, args.image_root, args.cache_root,
                               split="train", transform=None, crop_size=args.crop_size)
    val_ds = HaGRIDv2Dataset(args.ann_root, args.image_root, args.cache_root,
                             split="val", transform=None, crop_size=args.crop_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for crop, lm, label in loader:
        crop, lm, label = crop.to(device), lm.to(device), label.to(device)
        pred = model(crop, lm).argmax(1)
        correct += (pred == label).sum().item()
        total += label.numel()
    return correct / max(total, 1)


def main():
    args = parse_args()
    device = args.device
    model_cfg = {"crop_size": args.crop_size}

    train_loader, val_loader = build_loaders(args)
    model = build_model(model_cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for crop, lm, label in train_loader:
            crop, lm, label = crop.to(device), lm.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(crop, lm), label)
            loss.backward()
            optimizer.step()
            running += loss.item()

        val_acc = evaluate_acc(model, val_loader, device)
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"loss={running/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_cfg": model_cfg,
                    "label_map": LABEL_MAP,
                    "val_acc": val_acc,
                },
                out_dir / "gesture_model.pth",
            )
            print(f"  saved checkpoint (val_acc={val_acc:.4f})")

    print(f"done. best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()