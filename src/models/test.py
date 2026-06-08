#!/usr/bin/env python3
"""Baseline gesture model — fast to train, correct I/O.

Dual-branch design (kept identical to the eventual MobileNetV3 version):
    image branch    : tiny CNN  (later -> MobileNetV3-Small)
    landmark branch : wrist-relative normalize -> 42-d MLP
    fusion          : concat embeddings -> 6-class head

forward(crop, landmarks) -> logits (B, 6)
build_model(model_cfg)   -> nn.Module   (seam used by compression / predictor)
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 6  # 0=N/A, 1=fist, 2=like, 3=ok, 4=one, 5=palm


class _TinyCNN(nn.Module):
    """Placeholder image backbone. Swap for MobileNetV3-Small later."""

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),   # 112 -> 56
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 56 -> 28
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 28 -> 14
            nn.AdaptiveAvgPool2d(1),                                                                # -> (B,64,1,1)
        )
        self.out_dim = out_dim
        self.proj = nn.Identity() if out_dim == 64 else nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)  # (B, 64)
        return self.proj(x)


class _LandmarkMLP(nn.Module):
    """21x2 landmarks -> wrist-relative normalize -> 42-d -> embedding."""

    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        # landmarks: (B, 21, 2) — normalize relative to wrist (point 0), scale by span
        wrist = landmarks[:, 0:1, :]                      # (B, 1, 2)
        rel = landmarks - wrist                           # translation invariant
        span = rel.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        rel = rel / span                                  # scale invariant
        return self.net(rel.flatten(1))                   # (B, out_dim)


class GestureNet(nn.Module):
    def __init__(self, crop_size: int = 112, img_dim: int = 64, lm_dim: int = 32) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.image_branch = _TinyCNN(out_dim=img_dim)
        self.landmark_branch = _LandmarkMLP(out_dim=lm_dim)
        self.head = nn.Sequential(
            nn.Linear(img_dim + lm_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, crop: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        # crop expected already resized to (B, 3, crop_size, crop_size) by the
        # transform (train) or predictor letterbox (inference).
        img_emb = self.image_branch(crop)
        lm_emb = self.landmark_branch(landmarks)
        fused = torch.cat([img_emb, lm_emb], dim=1)
        return self.head(fused)


def build_model(model_cfg: dict | None = None) -> nn.Module:
    """Seam used by compression / predictor. cfg keys are all optional here."""
    cfg = model_cfg or {}
    return GestureNet(
        crop_size=cfg.get("crop_size", 112),
        img_dim=cfg.get("img_dim", 64),
        lm_dim=cfg.get("lm_dim", 32),
    )