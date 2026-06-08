"""
transforms.py

這裡定義 Data Augmentation 的主要流程，已對齊目前 dataset.py 的契約：

    transform(crop: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]

重要規則：
1. 輸入 / 輸出 crop 都是 uint8 RGB，不做 /255，也不做 ImageNet normalize。
2. 不做 letterbox resize；letterbox + normalize 交給 predictor.crop_to_input()。
3. 幾何變換一定同步更新 landmarks。
4. 光度變換只改 crop，不改 landmarks。
5. 只給 training 使用；val / test 請傳 transform=None。
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
    pixels_to_landmarks,
    validate_landmarks,
)


@dataclass
class AugmentationConfig:
    """
    將 YAML 或 dict 讀進來後轉成較好使用的 config 物件。

    注意：
    - image_size 保留只是為了相容舊 config；目前 augmentation 不會使用它。
    - letterbox resize 由 dataset.py 後面的 crop_to_input() 負責。
    """
    image_size: int = 112
    geometric: Optional[Dict[str, Any]] = None
    photometric: Optional[Dict[str, Any]] = None


class GestureAugmentation:
    """
    手勢分類專用 augmentation。

    符合 dataset.py 目前的 transform 契約：
        crop_aug, landmarks_aug = aug(crop, landmarks)

    Args:
        crop:
            np.ndarray, RGB, shape = (H, W, 3), dtype = uint8
        landmarks:
            np.ndarray, shape = (21, 2)，normalized x/y，crop-relative

    Returns:
        crop:
            np.ndarray, RGB, dtype = uint8，尺寸可變
        landmarks:
            np.ndarray, shape = (21, 2)，dtype = float32，範圍 clip 到 [0, 1]
    """

    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg
        self.geometric_cfg = cfg.geometric or {}
        self.photometric_cfg = cfg.photometric or {}
        self.photo_aug = self._build_photometric_aug()

    def __call__(self, crop: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        crop = self._ensure_rgb_uint8(crop)
        landmarks = validate_landmarks(landmarks)

        # 幾何增強：會改變座標系，所以 crop 和 landmarks 必須一起動。
        if self._rand_p(self.geometric_cfg.get("p", 0.8)):
            crop, landmarks = self._affine_transform(crop, landmarks)

        # bbox jitter：直接模擬 preprocessing bbox 有些偏移 / 裁切。
        # 這會改變 crop 座標系，所以 landmarks 也要重新換算。
        if self._rand_p(self.geometric_cfg.get("bbox_jitter_p", 0.3)):
            crop, landmarks = self._bbox_jitter(crop, landmarks)

        # 光度增強：只改 RGB 影像，不改 landmarks。
        crop = self.photo_aug(image=crop)["image"]

        # dataset.py 期望輸出仍是 uint8 RGB；landmarks 給模型前保持 float32。
        crop = self._ensure_rgb_uint8(crop)
        landmarks = clip_landmarks(landmarks)
        return crop, landmarks.astype(np.float32)

    def _affine_transform(self, crop: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        做平移、縮放、小角度旋轉，並同步更新 landmarks。

        可調整參數在 default.yaml 的 geometric 區塊：
        - rotate_limit：旋轉角度上限，建議 10~15 度，不要太大。
        - shift_limit：平移比例，0.10 代表最多移動寬/高的 10%。
        - scale_limit：縮放範圍，[0.90, 1.10] 代表 90%~110%。
        """
        h, w = crop.shape[:2]
        rotate_limit = float(self.geometric_cfg.get("rotate_limit", 12.0))
        shift_limit = float(self.geometric_cfg.get("shift_limit", 0.10))
        scale_min, scale_max = self.geometric_cfg.get("scale_limit", [0.90, 1.10])

        angle = np.random.uniform(-rotate_limit, rotate_limit)
        scale = np.random.uniform(float(scale_min), float(scale_max))
        tx = np.random.uniform(-shift_limit, shift_limit) * w
        ty = np.random.uniform(-shift_limit, shift_limit) * h

        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        # 黑邊只出現在 augmentation 訓練資料中，用來模擬 crop 失準。
        # 若黑邊太明顯，可把 borderValue 改成圖片平均色。
        warped = cv2.warpAffine(
            crop,
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

    def _bbox_jitter(self, crop: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        模擬 hand bbox jitter。

        做法：
        - 隨機調整 crop 的 left / right / top / bottom 邊界。
        - 輸出 crop 尺寸可能改變，這符合 dataset.py 的設計，因為後面會由
          predictor.crop_to_input() 統一 letterbox。
        - landmarks 會根據新的 crop 座標系重新正規化。

        可調整參數：
        - bbox_jitter_limit：邊界最多抖動多少比例，0.10 代表最多 10%。
        - bbox_jitter_p：發生機率。
        """
        h, w = crop.shape[:2]
        limit = float(self.geometric_cfg.get("bbox_jitter_limit", 0.10))
        limit = max(0.0, min(limit, 0.30))  # 避免裁切過度，導致 crop 太小。

        # 允許邊界往內裁，也允許往外擴。往外擴的部分用黑色 padding 補。
        dx1 = int(round(np.random.uniform(-limit, limit) * w))
        dx2 = int(round(np.random.uniform(-limit, limit) * w))
        dy1 = int(round(np.random.uniform(-limit, limit) * h))
        dy2 = int(round(np.random.uniform(-limit, limit) * h))

        # 新 bbox 在「原圖 crop 座標」中的位置；可能超出原圖範圍。
        x1 = dx1
        y1 = dy1
        x2 = w + dx2
        y2 = h + dy2

        # 防呆：避免新 crop 太小。
        min_w = max(8, int(w * 0.50))
        min_h = max(8, int(h * 0.50))
        if x2 - x1 < min_w or y2 - y1 < min_h:
            return crop, landmarks

        new_w = x2 - x1
        new_h = y2 - y1
        out = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        # 計算原 crop 與新 crop 的交集。
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)

        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return crop, landmarks

        out[dst_y1:dst_y2, dst_x1:dst_x2] = crop[src_y1:src_y2, src_x1:src_x2]

        pts = landmarks_to_pixels(landmarks, w, h)
        pts[:, 0] -= x1
        pts[:, 1] -= y1
        new_landmarks = pixels_to_landmarks(pts, new_w, new_h)
        return out, new_landmarks

    def _build_photometric_aug(self):
        """
        建立光度增強。

        這些變換只改 RGB crop，不改 landmarks：
        - ColorJitter：亮度 / 對比 / 飽和度 / 色相
        - GaussianBlur：模擬 TA blur / 手震 / 對焦不準
        - MotionBlur：模擬手部晃動
        - RandomBrightnessContrast：模擬低光或過曝
        - GaussNoise：模擬手機或 webcam 雜訊
        - ImageCompression：模擬壓縮失真
        """
        p_cfg = self.photometric_cfg
        transforms = []

        cj = p_cfg.get("color_jitter", {})
        if cj.get("enabled", True):
            transforms.append(
                A.ColorJitter(
                    brightness=float(cj.get("brightness", 0.25)),  # 越大，亮/暗變化越強
                    contrast=float(cj.get("contrast", 0.25)),      # 越大，陰影/強光變化越強
                    saturation=float(cj.get("saturation", 0.20)),  # 越大，膚色/背景色變化越強
                    hue=float(cj.get("hue", 0.05)),                # 不建議太大，避免顏色不自然
                    p=float(cj.get("p", 0.6)),
                )
            )

        low_light = p_cfg.get("low_light", {})
        if low_light.get("enabled", True):
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=float(low_light.get("brightness_limit", 0.30)),
                    contrast_limit=float(low_light.get("contrast_limit", 0.20)),
                    p=float(low_light.get("p", 0.25)),
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

    @staticmethod
    def _ensure_rgb_uint8(crop: np.ndarray) -> np.ndarray:
        """
        確保 crop 是 RGB uint8。

        注意：
        - 如果用 cv2.imread，會讀成 BGR，請在 dataset/preprocess 階段先轉 RGB。
        - 這裡不自動 BGR->RGB，避免重複轉換造成顏色錯誤。
        """
        crop = np.asarray(crop)
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(f"crop 應該是 shape (H, W, 3)，但收到 {crop.shape}")
        if crop.dtype != np.uint8:
            crop = np.clip(crop, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(crop)

    @staticmethod
    def _rand_p(p: float) -> bool:
        return np.random.rand() < float(p)
