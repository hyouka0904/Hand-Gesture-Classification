"""
build.py

從 YAML config 建立 augmentation pipeline。

使用方式：
    from src.augmentation import build_augmentation

    transform = build_augmentation("config/augmentation/default.yaml", train=True)
    image, landmarks, label = transform(image, landmarks, label)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .transforms import AugmentationConfig, GestureAugmentation


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """讀取 YAML 檔案。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 augmentation config：{path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def build_augmentation(cfg_path: str | Path, train: bool = True, override: Optional[Dict[str, Any]] = None):
    """
    建立 GestureAugmentation。

    Args:
        cfg_path:
            YAML 設定檔路徑，例如 config/augmentation/default.yaml
        train:
            True  = 啟用資料增強。
            False = 只做 letterbox resize，不做隨機增強。通常 validation/test 用 False。
        override:
            可選。用程式臨時覆蓋 YAML 參數。
            例如：override={"image_size": 224}

    Returns:
        GestureAugmentation instance
    """
    cfg_dict = load_yaml(cfg_path)
    if override:
        cfg_dict.update(override)

    cfg = AugmentationConfig(
        image_size=int(cfg_dict.get("image_size", 112)),
        train=bool(train),
        geometric=cfg_dict.get("geometric", {}),
        photometric=cfg_dict.get("photometric", {}),
        na_augmentation=cfg_dict.get("na_augmentation", {}),
        safety=cfg_dict.get("safety", {}),
    )
    return GestureAugmentation(cfg)
