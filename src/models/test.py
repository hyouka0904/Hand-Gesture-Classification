#!/usr/bin/env python3
"""Baseline gesture model.

Dual-branch classifier:

    crop image
    -> MobileNetV3-Small image branch
    -> image embedding
                         \
                          concat -> classifier head -> logits
                         /
    landmarks
    -> wrist-relative normalized landmark MLP
    -> landmark embedding

Forward contract:
    model(crop, landmarks) -> logits

Shapes:
    crop      : (B, 3, 112, 112)
    landmarks : (B, 21, 2)
    logits    : (B, 6)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

NUM_CLASSES = 6


class ImageBranch(nn.Module):
    def __init__(
        self,
        out_dim: int = 128,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        in_dim = backbone.classifier[0].in_features

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        x = self.features(crop)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class LandmarkBranch(nn.Module):
    def __init__(
        self,
        out_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(42, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        wrist = landmarks[:, 0:1, :]
        rel = landmarks - wrist

        span = rel.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        rel = rel / span

        return self.net(rel.flatten(1))


class GestureNet(nn.Module):
    def __init__(
        self,
        crop_size: int = 112,
        image_dim: int = 128,
        landmark_dim: int = 64,
        landmark_hidden_dim: int = 128,
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        self.crop_size = crop_size

        self.image_branch = ImageBranch(
            out_dim=image_dim,
            pretrained=pretrained,
            dropout=dropout,
        )

        self.landmark_branch = LandmarkBranch(
            out_dim=landmark_dim,
            hidden_dim=landmark_hidden_dim,
            dropout=dropout * 0.5,
        )

        self.head = nn.Sequential(
            nn.Linear(image_dim + landmark_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, NUM_CLASSES),
        )

    def forward(self, crop: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        image_emb = self.image_branch(crop)
        landmark_emb = self.landmark_branch(landmarks)
        fused = torch.cat([image_emb, landmark_emb], dim=1)
        return self.head(fused)


def build_model(model_cfg: dict[str, Any] | None = None) -> nn.Module:
    cfg = model_cfg or {}

    return GestureNet(
        crop_size=cfg.get("crop_size", 112),
        image_dim=cfg.get("image_dim", 128),
        landmark_dim=cfg.get("landmark_dim", 64),
        landmark_hidden_dim=cfg.get("landmark_hidden_dim", 128),
        head_hidden_dim=cfg.get("head_hidden_dim", 128),
        dropout=cfg.get("dropout", 0.2),
        pretrained=cfg.get("pretrained", False),
    )