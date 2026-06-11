#!/usr/bin/env python3
"""Evaluation driver.

Evaluates .pth or .onnx through GesturePredictor, matching the submission path.

Outputs:
    - confusion matrix
    - model size
    - plain accuracy
    - target accuracy
    - N/A false trigger rate
    - spec raw score
    - normalized score ratio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.predictor import GesturePredictor
from tqdm import tqdm

NUM_CLASSES = 6
LABEL_NAMES = ["N/A", "fist", "like", "ok", "one", "palm"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help=".pth or .onnx")
    p.add_argument("--data_root", default="data",
                   help="root holding processed/ (and mini_train/processed/)")
    p.add_argument("--mini_train", action="store_true",
                   help="read the mini subset cache under <data_root>/mini_train/processed")
    p.add_argument("--split", default="val")
    p.add_argument("--crop_size", type=int, default=112)
    p.add_argument("--conf_threshold", type=float, default=None,
                   help="override; default None -> .ptmodel uses its calibrated value, else 0.5")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def case_score(gt: int, pred: int) -> int:
    if gt != 0:
        return 1 if pred == gt else -2
    return -2 if pred != 0 else 0


def print_confusion(confusion: np.ndarray) -> None:
    print("GT\\Pred " + "".join(f"{n:>6}" for n in LABEL_NAMES))

    for i in range(NUM_CLASSES):
        row = "".join(f"{confusion[i, j]:>6}" for j in range(NUM_CLASSES))
        print(f"{LABEL_NAMES[i]:>7} {row}")


def compute_metrics(confusion: np.ndarray, raw_score: int, max_raw: int) -> dict[str, float]:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))

    target_total = int(confusion[1:, :].sum())
    target_correct = int(np.trace(confusion[1:, 1:]))

    na_total = int(confusion[0, :].sum())
    na_false_trigger = int(confusion[0, 1:].sum())

    return {
        "plain_accuracy": correct / max(total, 1),
        "target_accuracy": target_correct / max(target_total, 1),
        "na_false_trigger_rate": na_false_trigger / max(na_total, 1),
        "raw_score": float(raw_score),
        "max_raw_score": float(max_raw),
        "score_ratio": raw_score / max(max_raw, 1),
    }


def run_evaluate(
    weights_path: Path,
    data_root: str = "data",
    mini_train: bool = False,
    split: str = "val",
    crop_size: int = 112,
    conf_threshold: float | None = None,
    device: str = "cpu",
) -> dict:
    """Core evaluation logic. Returns metrics dict.

    model_builder is intentionally omitted: GesturePredictor auto-resolves
    build_model from meta['model_module'] baked into the .ptmodel at compress
    time. For .pth weights the caller should pass a model_builder via a
    GesturePredictor constructed outside — or simply use the .ptmodel path.
    """
    weights_path = Path(weights_path)
    model_size_mb = weights_path.stat().st_size / (1024 * 1024)

    predictor = GesturePredictor(
        weights_path=weights_path,
        crop_size=crop_size,
        conf_threshold=conf_threshold,
        device=device,
        # model_builder omitted -> auto-resolved from .ptmodel meta
    )

    if mini_train:
        cache_root = Path(data_root) / "mini_train" / "processed"
    else:
        cache_root = Path(data_root) / "processed"

    split_cache = cache_root / split
    samples = sorted(split_cache.rglob("*.npz"))

    if not samples:
        raise RuntimeError(f"No cached samples under {split_cache}")

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    raw_score = 0
    max_raw = 0

    for npz_path in tqdm(samples, desc="evaluating"):
        data = np.load(npz_path)
        crop = data["crop"]
        landmarks = data["landmarks"]
        gt = int(data["label"])
        pred = predictor.predict(crop, landmarks)
        confusion[gt, pred] += 1
        raw_score += case_score(gt, pred)
        if gt != 0:
            max_raw += 1

    metrics = compute_metrics(confusion, raw_score, max_raw)
    size_score = (10 - model_size_mb) * 3 if model_size_mb <= 10 else 0.0

    print(f"\nweights = {weights_path}")
    print(f"split = {split}")
    print(f"samples = {len(samples)}")
    print(f"model_size_mb = {model_size_mb:.4f}")
    print(f"conf_threshold = {predictor.conf_threshold:.4f}")

    print("\nConfusion matrix:")
    print_confusion(confusion)

    print("\nMetrics:")
    print(f"plain_accuracy = {metrics['plain_accuracy']:.4f}")
    print(f"target_accuracy = {metrics['target_accuracy']:.4f}")
    print(f"na_false_trigger_rate = {metrics['na_false_trigger_rate']:.4f}")
    print(f"RawScore = {int(metrics['raw_score'])}")
    print(f"MaxRawScore = {int(metrics['max_raw_score'])}")
    print(f"score_ratio = {metrics['score_ratio']:.4f}")
    print(f"Model Size         (max 30) = {size_score:.2f}")
    print(f"Basic Performance  (x20)    = {metrics['score_ratio'] * 20:.2f}")
    print(f"Robustness         (x40)    = {metrics['score_ratio'] * 40:.2f}")

    return metrics


def main() -> None:
    args = parse_args()
    run_evaluate(
        weights_path=Path(args.weights),
        data_root=args.data_root,
        mini_train=args.mini_train,
        split=args.split,
        crop_size=args.crop_size,
        conf_threshold=args.conf_threshold,
        device=args.device,
    )

if __name__ == "__main__":
    main()