#!/usr/bin/env python3
"""Baseline training driver.

Builds DataLoaders from the cached HaGRIDv2 dataset, trains the model, and saves
an .pth checkpoint — the fixed handoff format consumed by src/compression:

    {model_state_dict, model_cfg, label_map, val_acc}

Augmentation note:
- Training split can use src.augmentation through --aug_cfg.
- Validation split always uses transform=None so validation stays consistent
  with inference preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import HaGRIDv2Dataset
from src.models.test import build_model   # baseline model; swap for real one later

try:
    from src.augmentation import build_augmentation
except ImportError:
    build_augmentation = None


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

    # Augmentation 設定檔。只會用在 train split，val/test 一律不做 augmentation。
    # 若不想使用 augmentation，可以傳 --aug_cfg none 或直接不填。
    p.add_argument(
        "--aug_cfg",
        default="config/augmentation/default.yaml",
        help="Path to augmentation YAML. Use 'none' to disable training augmentation.",
    )
    return p.parse_args()


def _build_train_transform(aug_cfg: Optional[str]):
    """建立 training augmentation；validation 不使用 augmentation。"""
    if aug_cfg is None or str(aug_cfg).lower() in {"", "none", "null", "false"}:
        print("[train] training augmentation disabled")
        return None

    if build_augmentation is None:
        raise ImportError(
            "無法 import src.augmentation.build_augmentation。"
            "請確認 src/augmentation/ 檔案存在，且 requirements-train.txt 已安裝。"
        )

    print(f"[train] using training augmentation config: {aug_cfg}")
    return build_augmentation(aug_cfg, train=True)


def build_loaders(args):
    train_transform = _build_train_transform(args.aug_cfg)

    train_ds = HaGRIDv2Dataset(
        args.ann_root,
        args.image_root,
        args.cache_root,
        split="train",
        transform=train_transform,
        crop_size=args.crop_size,
    )

    # validation 不做 augmentation，保持和 inference crop_to_input 一致。
    val_ds = HaGRIDv2Dataset(
        args.ann_root,
        args.image_root,
        args.cache_root,
        split="val",
        transform=None,
        crop_size=args.crop_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
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
              f"loss={running/max(len(train_loader), 1):.4f}  val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_cfg": model_cfg,
                    "label_map": LABEL_MAP,
                    "val_acc": val_acc,
                    "aug_cfg": None if str(args.aug_cfg).lower() in {"", "none", "null", "false"} else args.aug_cfg,
                },
                out_dir / "gesture_model.pth",
            )
            print(f"  saved checkpoint (val_acc={val_acc:.4f})")

    print(f"done. best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
