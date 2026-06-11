#!/usr/bin/env python3
"""Gesture model: MobileNetV3-Small image branch + hand-skeleton GCN landmark
branch, fused with cross-modal attention.

Motivated by recent (2024-2025) static hand-gesture work:
  - landmark structure is better modelled as a hand-skeleton *graph* (GCN over
    MediaPipe's 21-keypoint topology) than as a flat MLP, since it captures
    finger-joint connectivity;
  - image + landmark fusion benefits from *cross-attention* rather than plain
    concatenation: the landmark embedding queries the image's spatial tokens,
    letting hand geometry guide which image regions matter.

Design constraints kept in mind:
  - All learnable weights live in Conv2d / Linear layers, so the Deep
    Compression pipeline (global magnitude pruning + k-means weight sharing,
    which only touches Conv2d/Linear) can compress the whole network. The
    skeleton adjacency is a fixed buffer (no params); LayerNorm affine params
    are stored full-precision in the .ptmodel (kept as "raw" tensors).
  - Lightweight by construction; the compressed .ptmodel stays well under 10 MB.

Forward contract (unchanged from the baseline, drop-in compatible):
    model(crop, landmarks) -> logits

Shapes:
    crop      : (B, 3, crop_size, crop_size)   crop_size default 112
    landmarks : (B, 21, 2)                      crop-relative normalized [0, 1]
    logits    : (B, 6)

build_model(model_cfg) accepts (with sensible defaults / aliases):
    crop_size            int   (default 112)
    img_dim              int   (default 128)   image token / embedding dim
                               alias: image_dim
    lm_dim               int   (default 64)    landmark embedding dim
                               alias: landmark_dim
    gcn_hidden           int   (default 64)    GCN hidden width
    attn_dim             int   (default 128)   cross-attention dim
    head_hidden_dim      int   (default 128)
    dropout              float (default 0.2)
    pretrained           bool  (default False) ImageNet weights for the backbone
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

NUM_CLASSES = 6
NUM_LANDMARKS = 21

# MediaPipe Hands 21-keypoint skeleton edges (undirected).
#   0: wrist
#   1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm closure
]


def _normalized_adjacency() -> torch.Tensor:
    """Symmetric-normalized adjacency  Â = D^-1/2 (A + I) D^-1/2  for the hand
    skeleton. Returned as a (21, 21) float tensor (a fixed, non-learnable buffer)."""
    A = torch.zeros(NUM_LANDMARKS, NUM_LANDMARKS)
    for i, j in HAND_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(NUM_LANDMARKS)          # self-loops
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


# ── image branch: MobileNetV3-Small -> spatial tokens + global vector ─────────

class ImageBranch(nn.Module):
    """Outputs both a set of spatial tokens (for cross-attention K/V) and a
    pooled global embedding."""

    def __init__(self, out_dim: int = 128, pretrained: bool = False,
                 dropout: float = 0.2) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        self.features = backbone.features            # (B, 576, h, w)
        in_ch = backbone.classifier[0].in_features   # 576

        # 1x1 conv projects every spatial location to out_dim -> tokens.
        self.proj = nn.Conv2d(in_ch, out_dim, kernel_size=1)
        self.act = nn.Hardswish(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, crop: torch.Tensor):
        feat = self.features(crop)                   # (B, 576, h, w)
        feat = self.act(self.proj(feat))             # (B, out_dim, h, w)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)     # (B, H*W, out_dim)
        glob = feat.mean(dim=(2, 3))                 # (B, out_dim) global avg-pool
        glob = self.drop(glob)
        return tokens, glob


# ── landmark branch: hand-skeleton GCN ───────────────────────────────────────

class GCNLayer(nn.Module):
    """One graph-conv: H' = act( Â · H · W ). Â is a fixed buffer; W is Linear."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim), adj: (N, N)
        x = self.lin(x)                              # node-wise transform
        x = torch.einsum("ij,bjc->bic", adj, x)      # neighbourhood aggregation
        return F.relu(x, inplace=True)


class LandmarkBranch(nn.Module):
    def __init__(self, out_dim: int = 64, gcn_hidden: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("adj", _normalized_adjacency())  # (21, 21)

        # per-node input feature: wrist-relative (x, y) = 2 dims
        self.gcn1 = GCNLayer(2, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, out_dim)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, landmarks: torch.Tensor):
        # wrist-relative, scale-normalized coordinates
        wrist = landmarks[:, 0:1, :]
        rel = landmarks - wrist
        span = rel.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        rel = rel / span                             # (B, 21, 2)

        h = self.gcn1(rel, self.adj)                 # (B, 21, gcn_hidden)
        h = self.gcn2(h, self.adj)                   # (B, 21, out_dim)
        h = self.drop(h)

        nodes = h                                    # (B, 21, out_dim) per-node
        glob = h.mean(dim=1)                         # (B, out_dim) graph readout
        return nodes, glob


# ── cross-modal attention fusion ─────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """Landmark embedding queries the image spatial tokens (single-head).

        Q = Wq(landmark_global)      (B, 1, d)
        K = Wk(image_tokens)         (B, N, d)
        V = Wv(image_tokens)         (B, N, d)
        ctx = softmax(QKᵀ/√d) V      (B, d)

    The fused vector is [image-context ‖ landmark_global], then projected."""

    def __init__(self, img_dim: int, lm_dim: int, attn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.q = nn.Linear(lm_dim, attn_dim)
        self.k = nn.Linear(img_dim, attn_dim)
        self.v = nn.Linear(img_dim, attn_dim)
        self.norm_q = nn.LayerNorm(lm_dim)
        self.norm_kv = nn.LayerNorm(img_dim)
        self.out = nn.Linear(attn_dim + lm_dim, attn_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = attn_dim ** -0.5
        self.out_dim = attn_dim

    def forward(self, img_tokens: torch.Tensor, lm_global: torch.Tensor):
        # img_tokens: (B, N, img_dim), lm_global: (B, lm_dim)
        q = self.q(self.norm_q(lm_global)).unsqueeze(1)   # (B, 1, d)
        kv_in = self.norm_kv(img_tokens)
        k = self.k(kv_in)                                 # (B, N, d)
        v = self.v(kv_in)                                 # (B, N, d)

        attn = (q @ k.transpose(1, 2)) * self.scale       # (B, 1, N)
        attn = attn.softmax(dim=-1)
        ctx = (attn @ v).squeeze(1)                       # (B, d)

        fused = torch.cat([ctx, lm_global], dim=1)        # (B, d + lm_dim)
        fused = self.drop(F.relu(self.out(fused), inplace=True))
        return fused                                      # (B, d)


# ── full model ───────────────────────────────────────────────────────────────

class GestureNet(nn.Module):
    def __init__(
        self,
        crop_size: int = 112,
        img_dim: int = 128,
        lm_dim: int = 64,
        gcn_hidden: int = 64,
        attn_dim: int = 128,
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size

        self.image_branch = ImageBranch(
            out_dim=img_dim, pretrained=pretrained, dropout=dropout,
        )
        self.landmark_branch = LandmarkBranch(
            out_dim=lm_dim, gcn_hidden=gcn_hidden, dropout=dropout * 0.5,
        )
        self.fusion = CrossAttentionFusion(
            img_dim=img_dim, lm_dim=lm_dim, attn_dim=attn_dim, dropout=dropout * 0.5,
        )

        self.head = nn.Sequential(
            nn.Linear(attn_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, NUM_CLASSES),
        )

    def forward(self, crop: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        img_tokens, _img_glob = self.image_branch(crop)
        _lm_nodes, lm_glob = self.landmark_branch(landmarks)
        fused = self.fusion(img_tokens, lm_glob)
        return self.head(fused)


def build_model(model_cfg: dict[str, Any] | None = None) -> nn.Module:
    cfg = model_cfg or {}

    # accept both new (img_dim/lm_dim) and legacy (image_dim/landmark_dim) keys
    img_dim = cfg.get("img_dim", cfg.get("image_dim", 128))
    lm_dim = cfg.get("lm_dim", cfg.get("landmark_dim", 64))

    return GestureNet(
        crop_size=cfg.get("crop_size", 112),
        img_dim=img_dim,
        lm_dim=lm_dim,
        gcn_hidden=cfg.get("gcn_hidden", 64),
        attn_dim=cfg.get("attn_dim", 128),
        head_hidden_dim=cfg.get("head_hidden_dim", 128),
        dropout=cfg.get("dropout", 0.2),
        pretrained=cfg.get("pretrained", False),
    )