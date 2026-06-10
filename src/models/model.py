#!/usr/bin/env python3
"""
gesture_model_final.py
正式設計的雙分支手勢分類模型：
  - Image Branch    : PyTorch 官方預訓練的 MobileNetV3-Small (自動調整特徵維度)
  - Landmark Branch : 平移與縮放不變性的 42維 關節點 MLP
  - Fusion Head     : Concat 融合後的 6類 分類器
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 6  # 0=N/A, 1=fist, 2=like, 3=ok, 4=one, 5=palm

class _LandmarkMLP(nn.Module):
    """
    關節點特徵分支 (保持與測試版相同的平移與縮放不變性防禦機制)
    """
    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21 * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        # landmarks shape: (B, 21, 2)
        # 減去手腕點 (第0點)，達成平移不變性
        rel = landmarks - landmarks[:, 0:1, :]
        # 除以最大展開寬度，達成縮放不變性
        span = rel.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        rel = rel / span
        return self.net(rel.flatten(1))  # (B, out_dim)

class FinalGestureNet(nn.Module):
    def __init__(self, crop_size: int = 112, img_dim: int = 128, lm_dim: int = 32) -> None:
        super().__init__()
        self.crop_size = crop_size
        
        # 1. 影像分支：換成正式的 MobileNetV3-Small
        # 由於你們輸入是 112x112，我們不使用 weights=None 的完全隨機初始化，
        # 改用預訓練權重 (MobileNetV3_Small_Weights.DEFAULT) 可以大幅加快收斂速度！
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        base_mobilenet = models.mobilenet_v3_small(weights=weights)
        
        # 提取特徵層 (移除原本最後的重分類 Classifier 區塊)
        self.image_branch = base_mobilenet.features
        
        # MobileNetV3-Small 的 features 最後輸出的通道數是 576
        # 我們用一個 AdaptiveAvgPool2d 確保不論輸入大小，出來都是 (B, 576, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 將 576 維度映射到指定的 img_dim (預設 128)
        self.img_projector = nn.Sequential(
            nn.Linear(576, img_dim),
            nn.BatchNorm1d(img_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. 關節點分支
        self.landmark_branch = _LandmarkMLP(out_dim=lm_dim)
        
        # 3. 決策分類頭
        self.head = nn.Sequential(
            nn.Linear(img_dim + lm_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, crop: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        # crop shape: (B, 3, 112, 112)
        
        # 影像特徵提取
        img_features = self.image_branch(crop)  # -> (B, 576, 4, 4)
        img_features = self.pool(img_features).flatten(1)  # -> (B, 576)
        img_emb = self.img_projector(img_features)  # -> (B, img_dim)
        
        # 關節點特徵提取
        lm_emb = self.landmark_branch(landmarks)  # -> (B, lm_dim)
        
        # 特徵融合與分類
        fused = torch.cat([img_emb, lm_emb], dim=1)
        return self.head(fused)

def build_model(model_cfg: dict) -> nn.Module:
    """
    標準卡槽工廠函式，完美對齊 train.py 與 compress.py 的呼叫規格！
    """
    crop_size = model_cfg.get("crop_size", 112)
    img_dim = model_cfg.get("img_dim", 128)  # 正式模型特徵給大一點
    lm_dim = model_cfg.get("lm_dim", 32)
    return FinalGestureNet(crop_size=crop_size, img_dim=img_dim, lm_dim=lm_dim)