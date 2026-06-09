#!/usr/bin/env python3
"""Shared inference core: load weights + crop letterbox + N/A heuristic.

Auto-detects weight format by extension:
    .onnx     -> onnxruntime  (lightweight submission path)
    .pth      -> torch        (dev/eval path; needs build_model + model_cfg)
    .ptmodel  -> torch        (Deep Compression archive; decode -> build_model)

predict(crop, landmarks) -> int in {0..5}, with the N/A engineering heuristic.

conf_threshold resolution order:
    explicit arg  >  calibrated value baked into .ptmodel meta  >  0.5
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

NA_CLASS = 0
NUM_CLASSES = 6
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Crop preprocessing (shared by train transform & inference) ───────────────

def letterbox_resize(crop: np.ndarray, size: int) -> np.ndarray:
    """Resize (H,W,3) uint8 to (size,size,3), preserving aspect ratio with padding."""
    h, w = crop.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def crop_to_input(crop: np.ndarray, size: int) -> np.ndarray:
    """(H,W,3) uint8 RGB -> (1,3,size,size) float32, ImageNet-normalized."""
    img = letterbox_resize(crop, size).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = img.transpose(2, 0, 1)[None]  # (1,3,S,S)
    return np.ascontiguousarray(img, dtype=np.float32)


def landmarks_to_input(landmarks: np.ndarray) -> np.ndarray:
    """(21,2) float32 -> (1,21,2) float32. (wrist-normalize happens in model.)"""
    return np.ascontiguousarray(landmarks[None], dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── N/A heuristic ────────────────────────────────────────────────────────────

def landmark_gate(landmarks: np.ndarray) -> bool:
    """Return True if hand geometry looks valid."""
    # TODO(N/A heuristic): implement real geometric rejection here
    #   (finger joint angles / fingertip distances / span sanity check).
    #   Currently a no-op pass-through — N/A relies only on conf_threshold.
    return True


def decide_class(probs: np.ndarray, landmarks: np.ndarray, conf_threshold: float) -> int:
    """Map softmax probs -> final class with conservative N/A bias."""
    pred = int(probs.argmax())
    conf = float(probs[pred])

    if pred == NA_CLASS:
        return NA_CLASS
    if conf < conf_threshold:          # low confidence -> N/A (avoid -2 penalty)
        return NA_CLASS
    if not landmark_gate(landmarks):   # implausible geometry -> N/A
        return NA_CLASS
    return pred


# ── Predictor ────────────────────────────────────────────────────────────────

class GesturePredictor:
    def __init__(
        self,
        weights_path: str | Path,
        crop_size: int = 112,
        conf_threshold: Optional[float] = None,
        model_builder: Optional[Callable[[dict], "object"]] = None,
        device: str = "cpu",
    ) -> None:
        self.crop_size = crop_size
        self.device = device

        weights_path = Path(weights_path)
        self.backend = weights_path.suffix.lower()

        meta_threshold: Optional[float] = None

        if self.backend == ".onnx":
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(weights_path),
                                                  providers=["CPUExecutionProvider"])
            self._onnx_inputs = [i.name for i in self._session.get_inputs()]  # ["crop","landmarks"]

        elif self.backend == ".pth":
            import torch
            if model_builder is None:
                raise ValueError(
                    ".pth backend requires model_builder=fn(model_cfg)->nn.Module. "
                    "Pass it explicitly (do not hardcode the model module)."
                )
            ckpt = torch.load(weights_path, map_location=device)
            model = model_builder(ckpt.get("model_cfg", {}))
            model.load_state_dict(ckpt["model_state_dict"])
            self._model = model.to(device).eval()
            self._torch = torch

        elif self.backend == ".ptmodel":
            import torch
            if model_builder is None:
                raise ValueError(
                    ".ptmodel backend requires model_builder=fn(model_cfg)->nn.Module. "
                    "Pass it explicitly (do not hardcode the model module)."
                )
            # Lazy import: only a .ptmodel load pulls in the decoder.
            from src.compression.baseline import load_ptmodel
            model_cfg, state_dict, _label_map, meta = load_ptmodel(weights_path)
            model = model_builder(model_cfg)
            model.load_state_dict(state_dict)
            self._model = model.to(device).eval()
            self._torch = torch
            meta_threshold = meta.get("best_conf_threshold")

        else:
            raise ValueError(f"Unsupported weights format: {weights_path.suffix}")

        # resolve threshold: explicit arg > calibrated meta > 0.5
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
        elif meta_threshold is not None:
            self.conf_threshold = float(meta_threshold)
        else:
            self.conf_threshold = 0.5

    def _forward(self, crop_in: np.ndarray, lm_in: np.ndarray) -> np.ndarray:
        """Return logits (NUM_CLASSES,)."""
        if self.backend == ".onnx":
            out = self._session.run(
                None,
                {self._onnx_inputs[0]: crop_in, self._onnx_inputs[1]: lm_in},
            )[0]
            return out[0]
        else:
            t = self._torch
            with t.no_grad():
                logits = self._model(
                    t.from_numpy(crop_in).to(self.device),
                    t.from_numpy(lm_in).to(self.device),
                )
            return logits.cpu().numpy()[0]

    def predict(self, cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
        crop_in = crop_to_input(cropped_img, self.crop_size)
        lm_in = landmarks_to_input(landmarks)
        logits = self._forward(crop_in, lm_in)
        probs = _softmax(logits)
        return decide_class(probs, landmarks, self.conf_threshold)