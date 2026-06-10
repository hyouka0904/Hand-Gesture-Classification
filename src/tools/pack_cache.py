#!/usr/bin/env python3
"""Pack per-sample .npz cache into large memory-mappable .npy arrays.

Each .npz stores a raw variable-size crop ((H,W,3) uint8) + landmarks ((21,2) float32)
+ label. Opening hundreds of thousands of tiny compressed .npz files per epoch is the
training I/O bottleneck on Windows. This packer letterbox-resizes every crop ONCE to a
fixed (S,S,3) uint8 and concatenates each split into three .npy files that the trainer
memory-maps. Normalization is deferred to PackedHaGRIDDataset.__getitem__ so the on-disk
footprint stays uint8 (~1/4 of storing normalized float32).

Input layout:
    <cache_root>/<split>/**/*.npz

Output layout:
    <cache_root>/<split>_crops.npy        (N, S, S, 3) uint8
    <cache_root>/<split>_landmarks.npy    (N, 21, 2)   float32
    <cache_root>/<split>_labels.npy       (N,)         int64

CLI:
    python -m src.tools.pack_cache --cache_root data/processed --splits train val
    python -m src.tools.pack_cache --cache_root data/processed --splits train val --num_workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.predictor import letterbox_resize


# ── Worker (must be top-level for Windows spawn) ──────────────────────────────

def _pack_one(args: tuple[int, str, int]) -> int:
    """Read one .npz, letterbox the crop, write into the three mmap arrays at index i.

    Returns i so the caller can update the progress bar.
    Each mmap is opened independently per worker call (cheap: just re-maps the file).
    Writing to distinct indices has no race condition.
    """
    i, path_str, crop_size = args
    # Re-open mmap arrays inside the worker (each worker opens its own file handle).
    # _mmap_paths is set as a module-level global by the initializer to avoid passing
    # large strings repeatedly through the task queue.
    crops_mm = np.load(_crops_path, mmap_mode="r+")
    lm_mm = np.load(_lm_path, mmap_mode="r+")
    labels_mm = np.load(_labels_path, mmap_mode="r+")

    data = np.load(path_str)
    crops_mm[i] = letterbox_resize(data["crop"], crop_size)
    lm_mm[i] = data["landmarks"].astype(np.float32, copy=False)
    labels_mm[i] = int(data["label"])

    crops_mm.flush()
    lm_mm.flush()
    labels_mm.flush()

    return i


# Module-level globals set by worker initializer (avoids pickling paths per task).
_crops_path: str = ""
_lm_path: str = ""
_labels_path: str = ""


def _worker_init(crops_path: str, lm_path: str, labels_path: str) -> None:
    global _crops_path, _lm_path, _labels_path
    _crops_path = crops_path
    _lm_path = lm_path
    _labels_path = labels_path


# ── Main pack logic ───────────────────────────────────────────────────────────

def pack_split(cache_root: Path, split: str, crop_size: int, num_workers: int) -> None:
    split_dir = cache_root / split
    files = sorted(split_dir.rglob("*.npz"))  # deterministic order

    if not files:
        raise RuntimeError(f"No .npz files found under {split_dir}")

    print(f"[pack] split={split}  files={len(files)}  crop_size={crop_size}  workers={num_workers}")

    first = np.load(files[0])
    lm_shape = first["landmarks"].shape

    crops_path = cache_root / f"{split}_crops.npy"
    lm_path = cache_root / f"{split}_landmarks.npy"
    labels_path = cache_root / f"{split}_labels.npy"

    # Allocate mmap arrays (w+ creates / overwrites).
    crops = np.lib.format.open_memmap(
        crops_path, mode="w+", dtype=np.uint8,
        shape=(len(files), crop_size, crop_size, 3),
    )
    lm = np.lib.format.open_memmap(
        lm_path, mode="w+", dtype=np.float32,
        shape=(len(files), *lm_shape),
    )
    labels = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int64,
        shape=(len(files),),
    )
    # Flush to materialise the files on disk before workers open them.
    crops.flush(); lm.flush(); labels.flush()
    del crops, lm, labels  # workers will re-open via mmap

    tasks = [(i, str(f), crop_size) for i, f in enumerate(files)]

    if num_workers <= 1:
        # Single-process path (no spawn overhead, easier to debug).
        crops_mm = np.load(crops_path, mmap_mode="r+")
        lm_mm = np.load(lm_path, mmap_mode="r+")
        labels_mm = np.load(labels_path, mmap_mode="r+")
        for i, path_str, cs in tqdm(tasks, desc=f"pack/{split}"):
            data = np.load(path_str)
            crops_mm[i] = letterbox_resize(data["crop"], cs)
            lm_mm[i] = data["landmarks"].astype(np.float32, copy=False)
            labels_mm[i] = int(data["label"])
        crops_mm.flush(); lm_mm.flush(); labels_mm.flush()
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
            initargs=(str(crops_path), str(lm_path), str(labels_path)),
        ) as pool:
            futs = {pool.submit(_pack_one, t): t[0] for t in tasks}
            with tqdm(total=len(tasks), desc=f"pack/{split}") as bar:
                for fut in as_completed(futs):
                    fut.result()  # re-raise any worker exception
                    bar.update(1)

    print(f"[pack] saved {crops_path}")
    print(f"[pack] saved {lm_path}")
    print(f"[pack] saved {labels_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", default="data/processed")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--crop_size", type=int, default=112)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    for split in args.splits:
        pack_split(cache_root, split, args.crop_size, args.num_workers)


if __name__ == "__main__":
    main()