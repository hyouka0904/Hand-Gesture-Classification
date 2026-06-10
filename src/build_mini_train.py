#!/usr/bin/env python3
"""Build a balanced mini subset of HaGRIDv2 for fast iteration.

For each class folder under image_root, samples up to N images (default 2000),
splits them train/val (default 80/20), and writes annotation JSONs in the same
layout that src/dataset.py's _build_cache expects:

    <output_dir>/annotations/train/<class>.json
    <output_dir>/annotations/val/<class>.json

Each JSON maps {uuid: {}} — _build_cache only uses the uuid *keys* to locate
images under the (unchanged) image_root, so the value payload is intentionally
empty. This is just a subset of the original dataset; no images are copied.

Two-handed classes are skipped to match _build_cache's behavior.

Usable two ways:
    - imported:   from build_mini_train import build_mini_train
    - standalone: python -m src.build_mini_train --data_root data
              (reads <data_root>/hagridv2_512, writes <data_root>/mini_train)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Keep in sync with src/dataset.py
TWO_HANDED = {
    "hand_heart", "hand_heart2", "thumb_index2",
    "timeout", "holy", "take_picture", "xsign",
}
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def _list_uuids(class_dir: Path) -> list[str]:
    """Return image-file stems (uuids) under a class folder."""
    return [
        p.stem
        for p in sorted(class_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def build_mini_train(
    image_root: str | Path,
    output_dir: str | Path = "data/mini_train",
    per_class: int = 2000,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Sample a balanced mini subset and write train/val annotation JSONs.

    Args:
        image_root : hagridv2_512/ (contains per-class folders)
        output_dir : where to write annotations/ (default data/mini_train)
        per_class  : max images sampled per class (default 2000)
        val_ratio  : fraction of each class's sample placed in val (default 0.2)
        seed       : RNG seed for reproducible sampling/split

    Returns:
        Path to <output_dir>/annotations
    """
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    ann_train = output_dir / "annotations" / "train"
    ann_val = output_dir / "annotations" / "val"
    ann_train.mkdir(parents=True, exist_ok=True)
    ann_val.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    class_dirs = sorted(d for d in image_root.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {image_root}")

    total_train = total_val = 0
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name in TWO_HANDED:
            continue

        uuids = _list_uuids(class_dir)
        if not uuids:
            print(f"[mini] {class_name}: no images, skipping")
            continue

        rng.shuffle(uuids)
        sample = uuids[:per_class]

        n_val = int(round(len(sample) * val_ratio))
        val_uuids = sample[:n_val]
        train_uuids = sample[n_val:]

        with open(ann_train / f"{class_name}.json", "w") as f:
            json.dump({u: {} for u in train_uuids}, f)
        with open(ann_val / f"{class_name}.json", "w") as f:
            json.dump({u: {} for u in val_uuids}, f)

        total_train += len(train_uuids)
        total_val += len(val_uuids)
        print(f"[mini] {class_name}: {len(sample)} sampled "
              f"-> {len(train_uuids)} train / {len(val_uuids)} val")

    print(f"[mini] done. total {total_train} train / {total_val} val "
          f"-> {output_dir / 'annotations'}")
    return output_dir / "annotations"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        default="data",
        help="root containing hagridv2_512/; mini subset written to <data_root>/mini_train",
    )
    p.add_argument("--per_class", type=int, default=2000)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    build_mini_train(
        image_root=data_root / "hagridv2_512",
        output_dir=data_root / "mini_train",
        per_class=args.per_class,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()