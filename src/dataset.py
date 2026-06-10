#!/usr/bin/env python3
"""HaGRIDv2 dataset with MediaPipe preprocessing cache.

First run: runs MediaPipeHandPreprocessor on raw images, saves (crop, landmarks) as .npz cache.
Subsequent runs: loads cache directly, skips MediaPipe.

Two dataset classes:
    HaGRIDv2Dataset      — reads per-sample .npz (supports augmentation; builds cache if missing)
    PackedHaGRIDDataset  — reads split-level packed .npy via mmap (fast path, no augmentation)
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
from src.predictor import crop_to_input
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


# ── Dataset (npz, supports augmentation) ────────────────────────────────────────

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
        crop_size: int = 112,
    ) -> None:
        self.transform = transform
        self.crop_size = crop_size
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
        crop: np.ndarray = data["crop"]            # (H,W,3) uint8 RGB, variable size
        landmarks: np.ndarray = data["landmarks"]  # (21,2) float32, crop-relative [0,1]
        label: int = int(data["label"])

        # Augmentation (training only). Contract: (crop_u8, lm) -> (crop_u8, lm).
        # Geometric augs MUST move landmarks too. Runs BEFORE letterbox/normalize.
        if self.transform is not None:
            crop, landmarks = self.transform(crop, landmarks)

        # SAME preprocessing as inference (predictor.crop_to_input) — no train/serving skew.
        crop_in = crop_to_input(crop, self.crop_size)[0]   # (3, S, S) float32, ImageNet-normalized
        crop_tensor = torch.from_numpy(crop_in)
        landmarks_tensor = torch.from_numpy(
            np.ascontiguousarray(landmarks, dtype=np.float32)
        )                                                   # (21, 2)
        return crop_tensor, landmarks_tensor, label


# ── Packed dataset (mmap .npy, fast path, no augmentation) ───────────────────────

class PackedHaGRIDDataset(Dataset):
    """Fast training path: read a split's packed .npy cache via mmap.

    Expects the output of `python -m src.tools.pack_cache`:
        <cache_root>/<split>_crops.npy      (N, S, S, 3) uint8    letterboxed, NOT normalized
        <cache_root>/<split>_landmarks.npy  (N, 21, 2)   float32  crop-relative [0, 1]
        <cache_root>/<split>_labels.npy     (N,)         int64

    Why this exists: opening hundreds of thousands of tiny compressed .npz files per epoch
    is the Windows training I/O bottleneck. One mmap'd .npy per array removes the per-file
    open/decompress overhead so the DataLoader can keep the GPU fed.

    __getitem__ output is bit-identical to HaGRIDv2Dataset (without augmentation): the packed
    crop is already letterboxed, so crop_to_input's internal letterbox is a no-op and only the
    ImageNet normalize + transpose remain. Use ONLY when no augmentation is needed — packed
    crops cannot be geometrically augmented (landmarks would desync).
    """

    def __init__(self, cache_root: str | Path, split: str = "train") -> None:
        cache_root = Path(cache_root)
        crops_path = cache_root / f"{split}_crops.npy"
        landmarks_path = cache_root / f"{split}_landmarks.npy"
        labels_path = cache_root / f"{split}_labels.npy"

        for p in (crops_path, landmarks_path, labels_path):
            if not p.exists():
                raise FileNotFoundError(f"Packed cache missing: {p}")

        # mmap_mode='r': keep arrays on disk, page in on access (do not load whole split to RAM).
        self.crops = np.load(crops_path, mmap_mode="r")          # (N, S, S, 3) uint8
        self.landmarks = np.load(landmarks_path, mmap_mode="r")  # (N, 21, 2) float32
        self.labels = np.load(labels_path, mmap_mode="r")        # (N,) int64

        n = len(self.crops)
        if not (len(self.landmarks) == n == len(self.labels)):
            raise RuntimeError(
                f"Packed array length mismatch: crops={len(self.crops)} "
                f"landmarks={len(self.landmarks)} labels={len(self.labels)}"
            )

        self.crop_size = int(self.crops.shape[1])  # S, inferred from packed shape

        print(f"[packed_dataset] split='{split}' — {n} samples loaded from packed cache")

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Packed crop is letterboxed uint8 (S,S,3). crop_to_input's letterbox is a no-op here;
        # it applies /255 + ImageNet normalize + transpose -> (3, S, S) float32. This matches
        # HaGRIDv2Dataset exactly. crop_to_input copies (astype), so the readonly mmap is fine.
        crop_in = crop_to_input(self.crops[idx], self.crop_size)[0]  # (3, S, S) float32
        crop_tensor = torch.from_numpy(crop_in)
        landmarks_tensor = torch.from_numpy(
            np.array(self.landmarks[idx], dtype=np.float32, copy=True)  # copy: mmap view is readonly
        )                                                               # (21, 2)
        label = int(self.labels[idx])
        return crop_tensor, landmarks_tensor, label