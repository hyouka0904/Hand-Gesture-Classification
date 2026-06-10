#!/usr/bin/env python3
"""Submission entry point. TA harness calls: `from inference import predict`.

Fixed interface — do NOT rename the file or change predict()'s signature.
All paths are resolved relative to this file so it runs in a fresh Colab runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # make `src` importable regardless of cwd

from src.predictor import GesturePredictor
from src.models.test import build_model  # swap to the real model module at submission

# Default submission weights: Deep Compression archive (must be <= 10 MB).
_WEIGHTS = _HERE / "model" / "gesture_model.ptmodel"

_predictor: GesturePredictor | None = None


def _get_predictor() -> GesturePredictor:
    global _predictor
    if _predictor is None:
        print("[inference] loading model...", flush=True)
        _predictor = GesturePredictor(
            weights_path=_WEIGHTS,
            crop_size=112,
            conf_threshold=None,        # None -> use the threshold calibrated into the .ptmodel
            model_builder=build_model,  # caller picks the model module (predictor stays generic)
        )
        print("[inference] model ready.", flush=True)
    return _predictor


def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    Args:
        cropped_img : (H, W, 3) uint8 RGB, variable size (TA-cropped hand bbox).
        landmarks   : (21, 2) float32, crop-relative normalized [0, 1].
    Returns:
        int in {0=N/A, 1=fist, 2=like, 3=ok, 4=one, 5=palm}
    """
    return _get_predictor().predict(cropped_img, landmarks)