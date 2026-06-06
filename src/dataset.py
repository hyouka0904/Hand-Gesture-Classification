#!/usr/bin/env python3
"""HaGRIDv2 dataset with MediaPipe preprocessing cache.

First run: runs MediaPipeHandPreprocessor on raw images, saves (crop, landmarks) as .npz cache.
Subsequent runs: loads cache directly, skips MediaPipe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# hand_preprocess.py is at project root, not in src/
sys.path.insert(0, str(Path(__file__).parent.parent))
from hand_preprocess import MediaPipeHandPreprocessor

# ── Label mapping ──────────────────────────────────────────────────────────────
TARGET_CLASSES = {"fist": 1, "like": 2, "ok": 3, "one": 4, "palm": 5}
TWO_HANDED = {"hand_heart", "hand_heart2", "thumb_index2", "timeout", "holy", "take_picture", "xsign"}
NA_LABEL = 0

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_image(image_root: Path, class_name: str, uuid: str) -> Optional[Path]:
    folder = image_root / class_name
    for ext in IMAGE_EXTENSIONS:
        p = folder / f"{uuid}{ext}"
        if p.exists():
            return p
    return None


def _label_for_class(class_name: str) -> int:
    if class_name in TARGET_CLASSES:
        return TARGET_CLASSES[class_name]
    return NA_LABEL


def _build_cache(
    ann_root: Path,
    image_root: Path,
    cache_root: Path,
    split: str,
) -> None:
    """Run MediaPipe on all images for this split and save .npz cache."""
    split_ann = ann_root / split
    split_cache = cache_root / split
    split_cache.mkdir(parents=True, exist_ok=True)

    json_files = sorted(split_ann.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No annotation JSONs found in {split_ann}")

    with MediaPipeHandPreprocessor() as preprocessor:
        for json_file in json_files:
            class_name = json_file.stem
            if class_name in TWO_HANDED:
                continue

            with open(json_file) as f:
                annotations = json.load(f)

            label = _label_for_class(class_name)
            class_cache = split_cache / class_name
            class_cache.mkdir(exist_ok=True)

            for uuid, _ in tqdm(annotations.items(), desc=f"{split}/{class_name}"):
                cache_path = class_cache / f"{uuid}.npz"
                if cache_path.exists():
                    continue

                image_path = _find_image(image_root, class_name, uuid)
                if image_path is None:
                    continue

                result = preprocessor.preprocess_path(image_path)
                if result is None:
                    continue

                crop, landmarks = result
                np.savez_compressed(cache_path, crop=crop, landmarks=landmarks, label=label)


# ── Dataset ────────────────────────────────────────────────────────────────────

class HaGRIDv2Dataset(Dataset):
    """
    Args:
        ann_root    : path to annotations/ (contains train/ val/ test/)
        image_root  : path to hagridv2_512/ (contains per-class folders)
        cache_root  : where to store/read .npz cache (default: data/processed/)
        split       : 'train' | 'val' | 'test'
        transform   : albumentations transform applied to crop (image H×W×3)
    """

    def __init__(
        self,
        ann_root: str | Path,
        image_root: str | Path,
        cache_root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        cache_root = Path(cache_root)
        split_cache = cache_root / split

        # Build cache if missing
        if not split_cache.exists() or not any(split_cache.rglob("*.npz")):
            print(f"[dataset] Cache not found for split='{split}', building...")
            _build_cache(Path(ann_root), Path(image_root), cache_root, split)

        # Index all .npz files
        self.samples: list[Path] = sorted(split_cache.rglob("*.npz"))
        if not self.samples:
            raise RuntimeError(f"No cached samples found under {split_cache}")

        print(f"[dataset] split='{split}' — {len(self.samples)} samples loaded from cache")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        data = np.load(self.samples[idx])
        crop: np.ndarray = data["crop"]        # (H, W, 3) uint8 RGB
        landmarks: np.ndarray = data["landmarks"]  # (21, 2) float32
        label: int = int(data["label"])

        if self.transform is not None:
            augmented = self.transform(image=crop)
            crop = augmented["image"]

        # crop: (H, W, 3) uint8 → (3, H, W) float32 [0, 1]
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        landmarks_tensor = torch.from_numpy(landmarks)  # (21, 2) float32

        return crop_tensor, landmarks_tensor, label