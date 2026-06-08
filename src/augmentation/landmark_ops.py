"""
landmark_ops.py

這個檔案專門處理 MediaPipe hand landmarks。
本專案的 landmarks 格式固定為：
    shape = (21, 2)
    座標為 [0, 1]，且是相對 cropped hand image 的座標

注意：
- README / predictor 的介面都只使用 x, y，沒有 z。
- augmentation 做幾何變換時，會先把 normalized 座標轉成 pixel 座標，
  做完變換後再轉回 [0, 1]。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_EPS = 1e-6


def validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    檢查並轉換 landmarks 格式。

    Returns:
        np.ndarray, shape = (21, 2), dtype = float32

    可調整方向：
    - 目前嚴格要求 (21, 2)，這符合本專案 hand_preprocess / predictor 的契約。
    - 如果之後 annotation 來源變成攤平的 42 維，可以在這裡加 reshape 邏輯。
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.shape != (21, 2):
        raise ValueError(f"landmarks shape 應該是 (21, 2)，但收到 {landmarks.shape}")

    if not np.isfinite(landmarks).all():
        raise ValueError("landmarks 中含有 NaN 或 Inf")

    return landmarks


def landmarks_to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    將 normalized landmarks [0, 1] 轉成 pixel 座標。

    Args:
        landmarks: shape (21, 2)，x/y in [0, 1]
        width: crop width
        height: crop height

    Returns:
        shape (21, 2)，x/y 為 pixel 座標
    """
    landmarks = validate_landmarks(landmarks)
    pixels = landmarks.copy()
    pixels[:, 0] *= float(width)
    pixels[:, 1] *= float(height)
    return pixels.astype(np.float32)


def pixels_to_landmarks(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    將 pixel 座標轉回 normalized landmarks。

    注意：
    - 這裡不會自動 clip 到 [0, 1]。
    - 是否 clip 由 transform 最後統一處理，避免太早丟失出界資訊。
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (21, 2):
        raise ValueError(f"pixel points shape 應該是 (21, 2)，但收到 {points.shape}")

    out = points.copy()
    out[:, 0] /= max(float(width), _EPS)
    out[:, 1] /= max(float(height), _EPS)
    return out.astype(np.float32)


def clip_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    將 landmarks 限制在 [0, 1]。

    用途：
    - augmentation 最後輸出給 dataset.py 前使用。
    - dataset.py 之後會直接把 landmarks 餵給模型，所以保持合法範圍較安全。
    """
    landmarks = validate_landmarks(landmarks)
    return np.clip(landmarks, 0.0, 1.0).astype(np.float32)


def out_of_bounds_ratio(landmarks: np.ndarray, margin: float = 0.0) -> float:
    """
    計算 landmarks 有多少比例超出 crop 範圍。

    Args:
        margin:
            可調參數。允許 landmark 超出邊界多少。
            margin = 0.00：只要小於 0 或大於 1 就算出界。
            margin = 0.05：允許座標落在 [-0.05, 1.05]，較寬鬆。

    目前 augmentation 主流程不會改 label，因此這個函式主要留給未來
    predictor 的 N/A rule 或 dataset 的 hard-negative logic 使用。
    """
    landmarks = validate_landmarks(landmarks)
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    outside = (x < -margin) | (x > 1.0 + margin) | (y < -margin) | (y > 1.0 + margin)
    return float(outside.mean())


def wrist_relative_normalize(landmarks: np.ndarray) -> np.ndarray:
    """
    將 landmarks 轉成以 wrist，也就是第 0 點為原點的表示。

    用途：
    - 給 model 的 landmark branch 使用。
    - 讓 landmark 特徵比較不受手在 crop 中位置與大小影響。

    可調整方向：
    - 目前 scale 使用 max(x_span, y_span)。
    - 若之後想更精細，可以改成 wrist 到 middle fingertip 的距離。
    """
    landmarks = validate_landmarks(landmarks)
    centered = landmarks - landmarks[0:1]

    x_span = landmarks[:, 0].max() - landmarks[:, 0].min()
    y_span = landmarks[:, 1].max() - landmarks[:, 1].min()
    scale = max(float(x_span), float(y_span), _EPS)

    return (centered / scale).astype(np.float32)


def landmark_bbox(landmarks: np.ndarray, margin: float = 0.05) -> Tuple[float, float, float, float]:
    """
    根據 landmarks 算出 normalized bbox。

    Args:
        margin:
            可調參數。bbox 額外擴張比例。
            margin 越大，保留手周圍背景越多。
            margin 越小，crop 越貼近手部。

    Returns:
        x1, y1, x2, y2，範圍會 clip 到 [0, 1]
    """
    landmarks = validate_landmarks(landmarks)
    x1 = float(landmarks[:, 0].min() - margin)
    y1 = float(landmarks[:, 1].min() - margin)
    x2 = float(landmarks[:, 0].max() + margin)
    y2 = float(landmarks[:, 1].max() + margin)

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    return x1, y1, x2, y2
