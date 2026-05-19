#!/usr/bin/env python3
"""MediaPipe preprocessing used by the hand-gesture challenge.

The classifier should only receive:
1. the cropped hand image
2. 21 landmarks in crop-relative coordinates
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from PIL import Image, ImageOps

_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _default_model_path() -> Path:
    return Path(__file__).with_name(_MODEL_FILENAME)


def _ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return
    print(f"Downloading MediaPipe model to {model_path}", flush=True)
    urllib.request.urlretrieve(_MODEL_URL, model_path)


class MediaPipeHandPreprocessor:
    """Detect one hand in an image and return its crop and landmarks."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        padding: float = 0.3,
        max_num_hands: int = 1,
        model_path: str | Path | None = None,
    ) -> None:
        self.padding = padding
        model_path = Path(model_path) if model_path is not None else _default_model_path()
        _ensure_model(model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "MediaPipeHandPreprocessor":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def preprocess_path(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
        """Student-facing helper: load an image path and return (crop, landmarks)."""
        with Image.open(image_path) as image:
            return self.preprocess_image(image)

    def preprocess_image(self, image: Image.Image | np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Student-facing helper: pass a PIL image or RGB array and return (crop, landmarks)."""
        result = self.detect_hand(image)
        if result is None:
            return None
        crop, landmarks, _bbox, _image_landmarks, _num_hands = result
        return crop, landmarks

    def detect_hand(
        self,
        image: Image.Image | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], np.ndarray, int] | None:
        pil_image = to_rgb_image(image)
        width, height = pil_image.size

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.asarray(pil_image),
        )
        result = self._detector.detect(mp_image)
        detected_hands = result.hand_landmarks or []
        if not detected_hands:
            return None

        image_landmarks = np.array(
            [[point.x, point.y] for point in detected_hands[0]],
            dtype=np.float32,
        )
        bbox = landmark_bbox(image_landmarks, width, height, self.padding)
        crop = np.asarray(pil_image.crop(bbox), dtype=np.uint8)
        crop_landmarks = landmarks_relative_to_crop(image_landmarks, bbox, width, height)
        return crop, crop_landmarks, bbox, image_landmarks, len(detected_hands)


def to_rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return Image.fromarray(array[:, :, :3].astype(np.uint8)).convert("RGB")


def landmark_bbox(
    landmarks: np.ndarray,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x = np.clip(landmarks[:, 0], 0.0, 1.0) * image_width
    y = np.clip(landmarks[:, 1], 0.0, 1.0) * image_height

    left, right = float(x.min()), float(x.max())
    top, bottom = float(y.min()), float(y.max())
    pad = padding * max(right - left, bottom - top, 1.0)

    return clamp_box(
        left - pad,
        top - pad,
        right + pad,
        bottom + pad,
        image_width,
        image_height,
    )


def clamp_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left_i = max(0, min(image_width - 1, int(np.floor(left))))
    top_i = max(0, min(image_height - 1, int(np.floor(top))))
    right_i = max(left_i + 1, min(image_width, int(np.ceil(right))))
    bottom_i = max(top_i + 1, min(image_height, int(np.ceil(bottom))))
    return left_i, top_i, right_i, bottom_i


def landmarks_relative_to_crop(
    landmarks: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    left, top, right, bottom = bbox
    crop_width = right - left
    crop_height = bottom - top

    crop_landmarks = landmarks.copy()
    crop_landmarks[:, 0] = (crop_landmarks[:, 0] * image_width - left) / crop_width
    crop_landmarks[:, 1] = (crop_landmarks[:, 1] * image_height - top) / crop_height
    return crop_landmarks.astype(np.float32)
