"""
build.py

從 YAML config 或 dict 建立 augmentation transform。

目前對齊 dataset.py 的 transform 契約：
    transform(crop, landmarks) -> (crop, landmarks)

建議使用方式：
    from src.augmentation import build_augmentation

    train_transform = build_augmentation("config/augmentation/default.yaml", train=True)
    val_transform = None  # val/test 請不要使用 augmentation
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


def build_transform(aug_cfg: Dict[str, Any]) -> GestureAugmentation:
    """
    工廠函式：由 dict 建立 transform。

    這是 augmentation 分工的標準接縫：
        build_transform(aug_cfg: dict) -> transform

    回傳的 transform 介面固定為：
        crop_aug, landmarks_aug = transform(crop, landmarks)

    Args:
        aug_cfg:
            通常來自 config/augmentation/default.yaml。
            可調整 geometric / photometric 區塊控制增強強度。
    """
    cfg = AugmentationConfig(
        image_size=int(aug_cfg.get("image_size", 112)),  # 保留相容；目前不做 letterbox
        geometric=aug_cfg.get("geometric", {}),
        photometric=aug_cfg.get("photometric", {}),
    )
    return GestureAugmentation(cfg)


def build_augmentation(
    cfg_path: str | Path,
    train: bool = True,
    override: Optional[Dict[str, Any]] = None,
):
    """
    從 YAML 建立 GestureAugmentation。

    Args:
        cfg_path:
            YAML 設定檔路徑，例如 config/augmentation/default.yaml
        train:
            True  = 回傳 training augmentation。
            False = 回傳 None。val/test 應該使用 transform=None，和 inference 保持一致。
        override:
            可選。用程式臨時覆蓋 YAML 參數。
            例如：override={"geometric": {"rotate_limit": 8}}

    Returns:
        train=True  -> GestureAugmentation instance
        train=False -> None
    """
    if not train:
        return None

    cfg_dict = load_yaml(cfg_path)
    if override:
        # 淺層覆蓋：如果 override geometric，會覆蓋整個 geometric 區塊。
        # 若要細部覆蓋，建議直接改 YAML 或在呼叫前自行合併 dict。
        cfg_dict.update(override)

    return build_transform(cfg_dict)
