#!/usr/bin/env python3
"""Baseline training driver.

Builds DataLoaders from the cached HaGRIDv2 dataset, trains the model, and saves
an .pth checkpoint — the fixed handoff format consumed by src/compression:

    {model_state_dict, model_cfg, label_map, val_acc}

Paths: a single --data_root (default data/) holds everything. The three roots are
derived under it:
    image_root = <data_root>/hagridv2_512
    ann_root   = <data_root>/annotations
    cache_root = <data_root>/processed

Dataset selection (build_loaders) is fully automatic — no separate pack step needed:
- Packed .npy cache present   -> PackedHaGRIDDataset (mmap fast path).
- Only per-sample .npz present -> auto-pack -> PackedHaGRIDDataset.
- Nothing cached              -> build .npz via MediaPipe -> auto-pack -> PackedHaGRIDDataset.
- Training augmentation on    -> HaGRIDv2Dataset (.npz), since packed crops are already
                                 letterboxed and cannot be geometrically augmented.

Augmentation note:
- Training split can use src.augmentation through --aug_cfg.
- Validation split always uses transform=None so validation stays consistent
  with inference preprocessing (and is therefore always safe to pack).

Mini-train note:
- --mini_train samples a balanced subset (default 2000/class, 80/20) via
  build_mini_train, then redirects ann_root + cache_root under
  <data_root>/mini_train so ALL generated files (annotations, .npz, packed .npy)
  stay inside the mini dir and never touch the full-dataset cache. image_root is
  left unchanged — the mini annotations reference uuids that still live under the
  original images.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HaGRIDv2Dataset, PackedHaGRIDDataset, _build_cache
from src.models.test import build_model   # baseline model; swap for real one later
from src.tools.pack_cache import pack_split
from src.compression.baseline import calibrate_threshold

try:
    from src.augmentation import build_augmentation
except ImportError:
    build_augmentation = None


LABEL_MAP = {0: "N/A", 1: "fist", 2: "like", 3: "ok", 4: "one", 5: "palm"}

# Subdirectory layout under --data_root
IMAGE_SUBDIR = "hagridv2_512"
ANN_SUBDIR = "annotations"
CACHE_SUBDIR = "processed"
MINI_SUBDIR = "mini_train"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        default="data",
        help="root holding hagridv2_512/, annotations/, processed/ (and mini_train/).",
    )
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
        default="none",
        help="Path to augmentation YAML. Use 'none' to disable training augmentation.",
    )

    # Mini-train 設定。預設 False；開啟後抽樣每 class 2000 張、8:2 切，
    # 所有暫存檔（annotations / .npz / packed .npy）都放在 <data_root>/mini_train 下。
    p.add_argument(
        "--mini_train",
        action="store_true",
        help="Use a balanced mini subset (default 2000/class, 80/20) under <data_root>/mini_train.",
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


def _has_packed_cache(cache_root: str | Path, split: str) -> bool:
    """True if all three packed .npy files exist for this split."""
    root = Path(cache_root)
    return (
        (root / f"{split}_crops.npy").exists()
        and (root / f"{split}_landmarks.npy").exists()
        and (root / f"{split}_labels.npy").exists()
    )


def _has_npz_cache(cache_root: str | Path, split: str) -> bool:
    """True if the per-sample .npz cache for this split exists (any .npz present)."""
    split_dir = Path(cache_root) / split
    return split_dir.exists() and any(split_dir.rglob("*.npz"))


def _ensure_packed(args, split: str) -> None:
    """Guarantee <cache_root>/<split>_*.npy exist.

    Builds the .npz cache (MediaPipe) if missing, then packs it into .npy. No-op if the
    packed cache is already present.
    """
    if _has_packed_cache(args.cache_root, split):
        return

    cache_root = Path(args.cache_root)

    if not _has_npz_cache(cache_root, split):
        print(f"[train] no .npz cache for split='{split}' — building via MediaPipe "
              f"(one-time, can be slow)...")
        _build_cache(Path(args.ann_root), Path(args.image_root), cache_root, split)

    print(f"[train] packing split='{split}' -> .npy (workers={args.num_workers})...")
    pack_split(cache_root, split, args.crop_size, args.num_workers)


def build_loaders(args):
    train_transform = _build_train_transform(args.aug_cfg)

    # Train: without augmentation we want the packed fast path. The packed cache stores
    # already-letterboxed crops, which cannot be geometrically augmented, so when
    # augmentation is on we must use the npz dataset instead.
    if train_transform is None:
        _ensure_packed(args, "train")
        print("[train] using packed train cache")
        train_ds = PackedHaGRIDDataset(args.cache_root, split="train")
    else:
        print("[train] augmentation enabled -> using npz train cache / raw dataset")
        train_ds = HaGRIDv2Dataset(
            args.ann_root,
            args.image_root,
            args.cache_root,
            split="train",
            transform=train_transform,
            crop_size=args.crop_size,
        )

    # Validation never uses augmentation, so the packed cache is always safe.
    _ensure_packed(args, "val")
    print("[train] using packed val cache")
    val_ds = PackedHaGRIDDataset(args.cache_root, split="val")

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

    # Derive the three roots under --data_root.
    data_root = Path(args.data_root)
    args.image_root = str(data_root / IMAGE_SUBDIR)
    args.ann_root = str(data_root / ANN_SUBDIR)
    args.cache_root = str(data_root / CACHE_SUBDIR)

    # --mini_train: 先抽樣寫出 mini annotations，再把 ann_root / cache_root 重導到
    # <data_root>/mini_train 之下。image_root 維持不變（mini annotations 的 uuid 仍指向原圖）。
    if args.mini_train:
        from src.build_mini_train import build_mini_train
        mini_dir = data_root / MINI_SUBDIR
        print("[train] --mini_train enabled: building mini subset...")
        build_mini_train(
            image_root=args.image_root,
            output_dir=mini_dir,
            per_class=2000,
        )
        args.ann_root = str(mini_dir / ANN_SUBDIR)
        args.cache_root = str(mini_dir / CACHE_SUBDIR)
        print(f"[train] mini_train paths -> ann_root={args.ann_root} "
              f"cache_root={args.cache_root} (image_root={args.image_root} unchanged)")
    else:
        print(f"[train] paths -> image_root={args.image_root} "
              f"ann_root={args.ann_root} cache_root={args.cache_root}")

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
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for crop, lm, label in pbar:
            crop, lm, label = crop.to(device), lm.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(crop, lm), label)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_acc = evaluate_acc(model, val_loader, device)
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"loss={running/max(len(train_loader), 1):.4f}  val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_tau, calib = calibrate_threshold(model, val_loader, device)
            torch.save(
                {
                    "best_conf_threshold": best_tau,
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