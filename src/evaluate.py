#!/usr/bin/env python3
"""Evaluation driver: faithful spec scoring + confusion matrix.

Evaluates through GesturePredictor — the SAME path as submission, including the
N/A heuristic — sample by sample, exactly like the TA harness calls predict().

Spec scoring (per sample):
    GT 5-class, pred == GT      -> +1
    GT 5-class, pred wrong/NA   -> -2
    GT N/A,     pred 5-class    -> -2   (false trigger)
    GT N/A,     pred N/A        ->  0
RawScore = sum;  MaxRawScore = #(GT in 1..5);  score = RawScore / MaxRawScore
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.predictor import GesturePredictor
from src.models.test import build_model   # baseline; only used for .pth eval

NUM_CLASSES = 6
LABEL_NAMES = ["N/A", "fist", "like", "ok", "one", "palm"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help=".pth or .onnx")
    p.add_argument("--cache_root", default="data/processed")
    p.add_argument("--split", default="val")
    p.add_argument("--crop_size", type=int, default=112)
    p.add_argument("--conf_threshold", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def case_score(gt: int, pred: int) -> int:
    if gt != 0:                       # GT is a 5-class target
        return 1 if pred == gt else -2
    return -2 if pred != 0 else 0     # GT is N/A: false trigger = -2, else 0


def main():
    args = parse_args()

    predictor = GesturePredictor(
        weights_path=args.weights,
        crop_size=args.crop_size,
        conf_threshold=args.conf_threshold,
        model_builder=build_model,    # only consumed by the .pth backend
        device=args.device,
    )

    split_cache = Path(args.cache_root) / args.split
    samples = sorted(split_cache.rglob("*.npz"))
    if not samples:
        raise RuntimeError(f"No cached samples under {split_cache}")

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    raw_score = 0
    max_raw = 0

    for npz_path in samples:
        data = np.load(npz_path)
        crop = data["crop"]
        landmarks = data["landmarks"]
        gt = int(data["label"])

        pred = predictor.predict(crop, landmarks)
        confusion[gt, pred] += 1
        raw_score += case_score(gt, pred)
        if gt != 0:
            max_raw += 1

    print(f"\nEvaluated {len(samples)} samples from split='{args.split}'\n")

    # Confusion matrix (rows = GT, cols = Pred)
    print("GT\\Pred " + "".join(f"{n:>6}" for n in LABEL_NAMES))
    for i in range(NUM_CLASSES):
        print(f"{LABEL_NAMES[i]:>7} " +
              "".join(f"{confusion[i, j]:>6}" for j in range(NUM_CLASSES)))

    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    print(f"\nplain accuracy = {correct}/{total} = {correct/total:.4f}")

    ratio = raw_score / max_raw if max_raw > 0 else 0.0
    print(f"RawScore = {raw_score}   MaxRawScore = {max_raw}   ratio = {ratio:.4f}")
    print(f"  -> Basic Performance  (×20) = {ratio * 20:.2f}")
    print(f"  -> Robustness         (×40) = {ratio * 40:.2f}")


if __name__ == "__main__":
    main()