"""
transforms.py

這裡定義 Data Augmentation 的主要流程。

設計重點：
1. 幾何變換要同步更新 image 與 landmarks。
2. 顏色、模糊、雜訊只改 image，不改 landmarks。
3. target class 使用保守增強，N/A class 可以使用較強增強。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError as exc:
    raise ImportError(
        "需要安裝 albumentations。請確認 requirements-train.txt 已安裝。"
    ) from exc

from .landmark_ops import (
    clip_landmarks,
    landmarks_to_pixels,
    out_of_bounds_ratio,
    pixels_to_landmarks,
    validate_landmarks,
)


@dataclass
class AugmentationConfig:
    """
    將 YAML 讀進來後轉成較好使用的 config 物件。

    大部分參數都可以在 config/augmentation/default.yaml 調整。
    """
    image_size: int = 112
    train: bool = True
    geometric: Optional[Dict[str, Any]] = None
    photometric: Optional[Dict[str, Any]] = None
    na_augmentation: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None


class GestureAugmentation:
    """
    手勢分類專用 augmentation。

    呼叫方式：
        aug = GestureAugmentation(cfg)
        image, landmarks, label = aug(image, landmarks, label)

    Args:
        image:
            np.ndarray, RGB, shape = (H, W, 3), dtype 通常是 uint8
        landmarks:
            np.ndarray, shape = (21, 2)，normalized x/y
        label:
            int, 0 表示 N/A，1~5 表示 target classes
    """

    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg
        self.image_size = int(cfg.image_size)
        self.geometric_cfg = cfg.geometric or {}
        self.photometric_cfg = cfg.photometric or {}
        self.na_cfg = cfg.na_augmentation or {}
        self.safety_cfg = cfg.safety or {}

        self.photo_aug = self._build_photometric_aug()
        self.na_photo_aug = self._build_na_photometric_aug()

    def __call__(self, image: np.ndarray, landmarks: np.ndarray, label: int) -> Tuple[np.ndarray, np.ndarray, int]:
        image = self._ensure_rgb_uint8(image)
        landmarks = validate_landmarks(landmarks)
        label = int(label)

        if self.cfg.train:
            image, landmarks, label = self._apply_training_aug(image, landmarks, label)

        # 最後固定輸出大小，方便 model 使用。
        image, landmarks = self._letterbox_resize(image, landmarks, self.image_size)

        # 給模型前建議 clip，避免極端 augmentation 產生非法座標。
        landmarks = clip_landmarks(landmarks)
        return image, landmarks.astype(np.float32), label

    def _apply_training_aug(self, image: np.ndarray, landmarks: np.ndarray, label: int):
        is_na = label == 0

        # 幾何增強：shift / scale / rotate，必須同步改 landmarks。
        if self._rand_p(self.geometric_cfg.get("p", 0.8)):
            image, landmarks = self._affine_transform(image, landmarks, is_na=is_na)

        # N/A 額外增強：讓模型更常看到模糊、裁切不完整、非標準情境。
        # 注意：這只建議用在 label=0，避免 target 被訓練得過度寬鬆。
        if is_na and self.na_cfg.get("enabled", True):
            image, landmarks = self._apply_na_geometric_aug(image, landmarks)
            image = self.na_photo_aug(image=image)["image"]

        # 一般影像品質增強：低光、模糊、雜訊、JPEG 壓縮。
        image = self.photo_aug(image=image)["image"]

        # 安全檢查：如果 landmarks 嚴重出界，可以選擇改成 N/A。
        # 這個設計適合「被切到太多的手」或「非標準姿勢」不要硬判 target。
        max_oob = float(self.safety_cfg.get("target_to_na_oob_ratio", 0.35))
        margin = float(self.safety_cfg.get("oob_margin", 0.02))
        if label != 0 and out_of_bounds_ratio(landmarks, margin=margin) > max_oob:
            if self.safety_cfg.get("convert_bad_target_to_na", True):
                label = 0

        return image, landmarks, label

    def _affine_transform(self, image: np.ndarray, landmarks: np.ndarray, is_na: bool):
        h, w = image.shape[:2]

        # target 使用較小角度；N/A 可稍微強一點。
        rotate_limit = float(self.geometric_cfg.get("rotate_limit", 12.0))
        shift_limit = float(self.geometric_cfg.get("shift_limit", 0.10))
        scale_min, scale_max = self.geometric_cfg.get("scale_limit", [0.85, 1.15])

        if is_na:
            rotate_limit *= float(self.na_cfg.get("na_rotate_multiplier", 1.5))
            shift_limit *= float(self.na_cfg.get("na_shift_multiplier", 1.2))

        angle = np.random.uniform(-rotate_limit, rotate_limit)
        scale = np.random.uniform(float(scale_min), float(scale_max))
        tx = np.random.uniform(-shift_limit, shift_limit) * w
        ty = np.random.uniform(-shift_limit, shift_limit) * h

        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        # borderValue 使用黑色。若覺得黑邊太明顯，可改成圖片平均色。
        warped = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        pts = landmarks_to_pixels(landmarks, w, h)
        pts_h = np.concatenate([pts, np.ones((21, 1), dtype=np.float32)], axis=1)
        new_pts = pts_h @ matrix.T
        new_landmarks = pixels_to_landmarks(new_pts, w, h)
        return warped, new_landmarks

    def _apply_na_geometric_aug(self, image: np.ndarray, landmarks: np.ndarray):
        """
        只針對 N/A 使用的較強幾何增強。

        可調整方向：
        - severe_crop_p 越大，N/A 越常出現被裁切的手。
        - severe_crop_ratio 越大，裁掉越多畫面。
        - 太強可能導致訓練分布不自然，建議慢慢加。
        """
        if not self._rand_p(self.na_cfg.get("severe_crop_p", 0.15)):
            return image, landmarks

        h, w = image.shape[:2]
        crop_ratio = float(self.na_cfg.get("severe_crop_ratio", 0.12))

        # 隨機選一個方向裁切，模擬 bbox 邊緣切到手。
        side = np.random.choice(["left", "right", "top", "bottom"])
        cut_x = int(w * np.random.uniform(0.03, crop_ratio))
        cut_y = int(h * np.random.uniform(0.03, crop_ratio))

        x1, y1, x2, y2 = 0, 0, w, h
        if side == "left":
            x1 = cut_x
        elif side == "right":
            x2 = w - cut_x
        elif side == "top":
            y1 = cut_y
        else:
            y2 = h - cut_y

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return image, landmarks

        pts = landmarks_to_pixels(landmarks, w, h)
        pts[:, 0] -= x1
        pts[:, 1] -= y1
        new_h, new_w = cropped.shape[:2]
        new_landmarks = pixels_to_landmarks(pts, new_w, new_h)
        return cropped, new_landmarks

    def _build_photometric_aug(self):
        p_cfg = self.photometric_cfg
        transforms = []

        cj = p_cfg.get("color_jitter", {})
        if cj.get("enabled", True):
            transforms.append(
                A.ColorJitter(
                    brightness=float(cj.get("brightness", 0.25)),  # 亮度變化，越大越亮/暗不穩
                    contrast=float(cj.get("contrast", 0.25)),      # 對比變化，越大越容易模擬強光/陰影
                    saturation=float(cj.get("saturation", 0.20)),  # 飽和度變化，影響膚色與背景色
                    hue=float(cj.get("hue", 0.05)),                # 色相變化，不建議太大
                    p=float(cj.get("p", 0.6)),
                )
            )

        blur = p_cfg.get("gaussian_blur", {})
        if blur.get("enabled", True):
            k = blur.get("kernel_size", [3, 5])
            transforms.append(
                A.GaussianBlur(
                    blur_limit=tuple(k),
                    sigma_limit=tuple(blur.get("sigma", [0.1, 1.5])),
                    p=float(blur.get("p", 0.25)),
                )
            )

        motion = p_cfg.get("motion_blur", {})
        if motion.get("enabled", True):
            transforms.append(
                A.MotionBlur(
                    blur_limit=tuple(motion.get("blur_limit", [3, 7])),
                    p=float(motion.get("p", 0.10)),
                )
            )

        noise = p_cfg.get("noise", {})
        if noise.get("enabled", True):
            transforms.append(
                A.GaussNoise(
                    var_limit=tuple(noise.get("var_limit", [5.0, 30.0])),
                    p=float(noise.get("p", 0.15)),
                )
            )

        jpeg = p_cfg.get("jpeg", {})
        if jpeg.get("enabled", True):
            transforms.append(
                A.ImageCompression(
                    quality_lower=int(jpeg.get("quality_lower", 45)),
                    quality_upper=int(jpeg.get("quality_upper", 95)),
                    p=float(jpeg.get("p", 0.15)),
                )
            )

        return A.Compose(transforms)

    def _build_na_photometric_aug(self):
        """
        N/A 專用影像品質增強。

        目的：
        - 讓模型看到更多「不該觸發指令」的困難樣本。
        - 比一般 photometric augmentation 稍強。
        """
        if not self.na_cfg.get("enabled", True):
            return A.Compose([])

        return A.Compose([
            A.GaussianBlur(
                blur_limit=tuple(self.na_cfg.get("severe_blur_limit", [5, 9])),
                sigma_limit=tuple(self.na_cfg.get("severe_blur_sigma", [1.0, 3.0])),
                p=float(self.na_cfg.get("severe_blur_p", 0.10)),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=float(self.na_cfg.get("brightness_limit", 0.35)),
                contrast_limit=float(self.na_cfg.get("contrast_limit", 0.35)),
                p=float(self.na_cfg.get("brightness_contrast_p", 0.20)),
            ),
        ])

    def _letterbox_resize(self, image: np.ndarray, landmarks: np.ndarray, size: int):
        """
        等比例 resize + padding，不拉伸手的形狀。

        這比直接 resize 更適合手勢，因為手掌比例不會被壓扁。
        landmarks 也會同步轉換到 padding 後的新座標。
        """
        h, w = image.shape[:2]
        scale = min(size / max(w, 1), size / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2

        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        pts = landmarks_to_pixels(landmarks, w, h)
        pts[:, 0] = pts[:, 0] * scale + pad_x
        pts[:, 1] = pts[:, 1] * scale + pad_y
        new_landmarks = pixels_to_landmarks(pts, size, size)
        return canvas, new_landmarks

    @staticmethod
    def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
        """
        確保 image 是 RGB uint8。

        注意：
        - 如果你用 cv2.imread，讀進來會是 BGR，應該在 dataset.py 先轉 RGB。
        - 這裡不自動 BGR->RGB，避免重複轉換造成顏色錯誤。
        """
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image 應該是 shape (H, W, 3)，但收到 {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _rand_p(p: float) -> bool:
        return np.random.rand() < float(p)
