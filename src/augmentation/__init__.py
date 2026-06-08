"""
src.augmentation

對外主要使用：
    build_augmentation("config/augmentation/default.yaml", train=True)

或符合分工契約的：
    build_transform(aug_cfg)

回傳 transform 的介面固定為：
    crop_aug, landmarks_aug = transform(crop, landmarks)
"""

from .build import build_augmentation, build_transform, load_yaml
from .landmark_ops import (
    clip_landmarks,
    landmark_bbox,
    landmarks_to_pixels,
    out_of_bounds_ratio,
    pixels_to_landmarks,
    validate_landmarks,
    wrist_relative_normalize,
)
from .transforms import AugmentationConfig, GestureAugmentation

__all__ = [
    "build_augmentation",
    "build_transform",
    "load_yaml",
    "AugmentationConfig",
    "GestureAugmentation",
    "validate_landmarks",
    "landmarks_to_pixels",
    "pixels_to_landmarks",
    "clip_landmarks",
    "out_of_bounds_ratio",
    "wrist_relative_normalize",
    "landmark_bbox",
]
