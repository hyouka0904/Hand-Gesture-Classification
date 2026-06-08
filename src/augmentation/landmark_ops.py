"""
landmark_ops.py

這個檔案專門處理 MediaPipe hand landmarks。
本專案的 landmarks 格式固定為：
    shape = (21, 2)
    座標為 [0, 1]，且是相對 cropped hand image 的座標

README 已說明沒有 z 座標，所以這裡所有函式都只處理x, y。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


_EPS = 1e-6


def validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    檢查並轉換 landmarks 格式。

    可調整方向：
    - 如果之後資料來源不是 (21, 2)，可以在這裡擴充轉換邏輯。
    - 目前設計是嚴格檢查，避免訓練時吃到錯誤 annotation。
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
        width: image width
        height: image height

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
    將 pixel 座標轉回 normalized landmarks [0, 1]。

    注意：
    - 這裡不會自動 clip 到 [0, 1]，因為我們有時候需要知道 landmark 是否出界。
    - 是否裁切到 [0, 1] 由外層 transform 控制。
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

    可調整方向：
    - 如果你希望保留出界資訊給 N/A rule，就不要太早呼叫這個函式。
    - 如果只是要餵給 neural network，通常 clip 後比較安全。
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

    用途：
    - 嚴重出界通常代表 bbox 裁切不完整或手勢不可靠，可作為 N/A 依據。
    """
    landmarks = validate_landmarks(landmarks)
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    outside = (x < -margin) | (x > 1.0 + margin) | (y < -margin) | (y > 1.0 + margin)
    return float(outside.mean())


def wrist_relative_normalize(landmarks: np.ndarray) -> np.ndarray:
    """
    將 landmarks 轉成以 wrist，也就是第 0 點為原點的表示。

    做法：
    1. 所有點減掉 wrist 座標。
    2. 再除以整隻手的 span，讓尺度更穩定。

    用途：
    - 給 landmark branch 的 MLP 使用。
    - 增加 translation / scale invariance。

    可調整方向：
    - 目前 scale 使用 max(x_span, y_span)。
    - 若想更精細，可改成 palm size 或 wrist 到 middle fingertip 的距離。
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
