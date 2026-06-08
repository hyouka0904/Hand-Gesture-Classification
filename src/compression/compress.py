#!/usr/bin/env python3
"""Model compression: pruning + retrain, then export to ONNX.

Two entry points (to be decided at the team meeting):
  - compress_from_pth  : Approach A. Input .pth (state_dict + cfg), retrain in
                         PyTorch, export .onnx.  [recommended — clean]
  - compress_from_onnx : Approach B. Input .onnx, convert back to nn.Module via
                         onnx2torch, retrain, re-export .onnx.  [fragile]

Both share the same prune_and_retrain core; they differ ONLY in how the
trainable nn.Module is obtained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader


# ── Shared core ──────────────────────────────────────────────────────────────

def prune_and_retrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    amount: float = 0.5,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cuda",
) -> nn.Module:
    """Global magnitude unstructured pruning + fine-tune retrain.

    Args:
        amount : fraction of weights to prune globally (0.5 = 50%).
        epochs : retrain epochs after pruning.
    Returns:
        model with pruning made permanent (re-densified state_dict, zeros baked in).
    """
    model = model.to(device)

    # 1. Collect prunable params (Conv2d + Linear weights)
    params_to_prune = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    if not params_to_prune:
        raise RuntimeError("No Conv2d/Linear layers found to prune.")

    # 2. Global magnitude pruning (zeroed grads stay zero — Deep Compression)
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # 3. Retrain (pruning masks keep pruned weights at zero during fine-tune)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for crop, landmarks, label in train_loader:
            crop = crop.to(device)
            landmarks = landmarks.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(crop, landmarks)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running += loss.item()

        msg = f"[compress] retrain epoch {epoch + 1}/{epochs}  loss={running / len(train_loader):.4f}"
        if val_loader is not None:
            msg += f"  val_acc={_quick_acc(model, val_loader, device):.4f}"
        print(msg)

    # 4. Make pruning permanent (remove masks, bake zeros into weights)
    for module, name in params_to_prune:
        prune.remove(module, name)

    return model


@torch.no_grad()
def _quick_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for crop, landmarks, label in loader:
        crop, landmarks, label = crop.to(device), landmarks.to(device), label.to(device)
        pred = model(crop, landmarks).argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()
    return correct / max(total, 1)


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    crop_size: int = 112,
    device: str = "cuda",
) -> None:
    """Export trained model to ONNX (inference format for submission)."""
    model = model.to(device).eval()
    dummy_crop = torch.randn(1, 3, crop_size, crop_size, device=device)
    dummy_landmarks = torch.randn(1, 21, 2, device=device)

    torch.onnx.export(
        model,
        (dummy_crop, dummy_landmarks),
        str(output_path),
        input_names=["crop", "landmarks"],
        output_names=["logits"],
        dynamic_axes={"crop": {0: "batch"}, "landmarks": {0: "batch"}},
        opset_version=17,
    )
    print(f"[compress] exported ONNX -> {output_path}")


# ── Approach A: .pth in → retrain → .onnx out  (recommended) ─────────────────

def compress_from_pth(
    pth_path: str | Path,
    onnx_out: str | Path,
    model_builder: Callable[[dict], nn.Module],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    amount: float = 0.5,
    epochs: int = 5,
    device: str = "cuda",
) -> None:
    """
    Args:
        pth_path      : checkpoint dict with 'model_state_dict' + 'model_cfg'.
        model_builder : fn(model_cfg) -> nn.Module  (provided by src/models).
    """
    ckpt = torch.load(pth_path, map_location=device)
    model = model_builder(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state_dict"])

    model = prune_and_retrain(model, train_loader, val_loader,
                              amount=amount, epochs=epochs, device=device)
    export_onnx(model, onnx_out, device=device)


# ── Approach B: .onnx in → onnx2torch → retrain → .onnx out  (fragile) ───────

def compress_from_onnx(
    onnx_in: str | Path,
    onnx_out: str | Path,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    amount: float = 0.5,
    epochs: int = 5,
    device: str = "cuda",
) -> None:
    """
    Converts ONNX back to a trainable nn.Module via onnx2torch, then runs the
    SAME prune_and_retrain core. WARNING: onnx2torch round-trip can fail on
    unsupported ops, lose layer names, and break the forward(crop, landmarks)
    signature — that is the cost of using ONNX as a retrain input.
    """
    try:
        from onnx2torch import convert
    except ImportError as e:
        raise ImportError(
            "Approach B needs onnx2torch: pip install onnx2torch"
        ) from e

    model = convert(str(onnx_in))  # nn.Module, but forward signature may differ!

    model = prune_and_retrain(model, train_loader, val_loader,
                              amount=amount, epochs=epochs, device=device)
    export_onnx(model, onnx_out, device=device)