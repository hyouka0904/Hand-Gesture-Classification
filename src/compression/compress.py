#!/usr/bin/env python3
"""Model compression driver.

Current compression baseline:
    .pth checkpoint
    -> global magnitude unstructured pruning
    -> fine-tune retraining
    -> compressed .pth
    -> compressed .onnx

This is pruning + fine-tuning only.
It is not full Deep Compression yet because it does not implement
weight sharing or Huffman coding.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from src.dataset import HaGRIDv2Dataset

try:
    from src.augmentation import build_augmentation
except ImportError:
    build_augmentation = None


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but CUDA is not available.")

    return device


def load_model_builder(module_name: str) -> Callable[[dict], nn.Module]:
    module = importlib.import_module(module_name)

    if not hasattr(module, "build_model"):
        raise AttributeError(f"{module_name} must define build_model(model_cfg).")

    return module.build_model


def make_default_outputs(pth_in: str | Path, amount: float) -> tuple[Path, Path]:
    pth_in = Path(pth_in)

    amount_tag = int(round(amount * 100))
    stem = f"{pth_in.stem}_pruned{amount_tag}"

    pth_out = pth_in.with_name(f"{stem}.pth")
    onnx_out = Path("model") / f"{stem}.onnx"

    return pth_out, onnx_out


def build_loaders(
    ann_root: str | Path,
    image_root: str | Path,
    cache_root: str | Path,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    aug_cfg: Optional[str],
) -> tuple[DataLoader, DataLoader]:
    train_transform = None

    if aug_cfg and str(aug_cfg).lower() not in {"none", "null", "false", ""}:
        if build_augmentation is None:
            raise ImportError(
                "Cannot import src.augmentation.build_augmentation. "
                "Install training dependencies or pass --aug_cfg none."
            )

        train_transform = build_augmentation(aug_cfg, train=True)

    train_ds = HaGRIDv2Dataset(
        ann_root=ann_root,
        image_root=image_root,
        cache_root=cache_root,
        split="train",
        transform=train_transform,
        crop_size=crop_size,
    )

    val_ds = HaGRIDv2Dataset(
        ann_root=ann_root,
        image_root=image_root,
        cache_root=cache_root,
        split="val",
        transform=None,
        crop_size=crop_size,
    )

    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def collect_prunable_params(model: nn.Module) -> list[tuple[nn.Module, str]]:
    params = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    if not params:
        raise RuntimeError("No Conv2d or Linear weights found to prune.")

    return params


@torch.no_grad()
def quick_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()

    correct = 0
    total = 0

    for crop, landmarks, label in loader:
        crop = crop.to(device)
        landmarks = landmarks.to(device)
        label = label.to(device)

        logits = model(crop, landmarks)
        pred = logits.argmax(dim=1)

        correct += (pred == label).sum().item()
        total += label.numel()

    return correct / max(total, 1)


@torch.no_grad()
def global_sparsity(model: nn.Module) -> float:
    zero = 0
    total = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.detach()
            zero += (weight == 0).sum().item()
            total += weight.numel()

    return zero / max(total, 1)


def prune_and_retrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    amount: float,
    epochs: int,
    lr: float,
    device: str,
) -> nn.Module:
    model = model.to(device)

    params_to_prune = collect_prunable_params(model)

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    print(f"[compress] pruning amount={amount:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for crop, landmarks, label in train_loader:
            crop = crop.to(device)
            landmarks = landmarks.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(crop, landmarks)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        msg = (
            f"[compress] epoch {epoch + 1}/{epochs} "
            f"loss={running_loss / max(len(train_loader), 1):.4f}"
        )

        if val_loader is not None:
            msg += f" val_acc={quick_acc(model, val_loader, device):.4f}"

        print(msg)

    for module, name in params_to_prune:
        prune.remove(module, name)

    print(f"[compress] final global sparsity={global_sparsity(model):.4f}")

    return model


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    crop_size: int,
    device: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()

    dummy_crop = torch.randn(1, 3, crop_size, crop_size, device=device)
    dummy_landmarks = torch.randn(1, 21, 2, device=device)

    torch.onnx.export(
        model,
        (dummy_crop, dummy_landmarks),
        str(output_path),
        input_names=["crop", "landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "crop": {0: "batch"},
            "landmarks": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[compress] exported ONNX -> {output_path} ({size_mb:.2f} MB)")


def save_compressed_checkpoint(
    model: nn.Module,
    original_ckpt: dict,
    output_path: str | Path,
    amount: float,
    epochs: int,
    lr: float,
    val_acc: Optional[float],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = dict(original_ckpt)
    ckpt["model_state_dict"] = model.state_dict()
    ckpt["compression"] = {
        "method": "global_magnitude_unstructured_pruning",
        "amount": amount,
        "retrain_epochs": epochs,
        "retrain_lr": lr,
        "global_sparsity": global_sparsity(model),
    }

    if val_acc is not None:
        ckpt["compressed_val_acc"] = val_acc

    torch.save(ckpt, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[compress] saved compressed PTH -> {output_path} ({size_mb:.2f} MB)")


def compress_from_pth(
    pth_path: str | Path,
    pth_out: str | Path,
    onnx_out: str | Path,
    model_builder: Callable[[dict], nn.Module],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    amount: float,
    epochs: int,
    lr: float,
    device: str,
) -> nn.Module:
    device = resolve_device(device)

    ckpt = torch.load(pth_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain 'model_state_dict'.")

    model_cfg = ckpt.get("model_cfg", {})
    crop_size = int(model_cfg.get("crop_size", 112))

    model = model_builder(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    model = prune_and_retrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        amount=amount,
        epochs=epochs,
        lr=lr,
        device=device,
    )

    val_acc = quick_acc(model, val_loader, device) if val_loader is not None else None

    save_compressed_checkpoint(
        model=model,
        original_ckpt=ckpt,
        output_path=pth_out,
        amount=amount,
        epochs=epochs,
        lr=lr,
        val_acc=val_acc,
    )

    export_onnx(
        model=model,
        output_path=onnx_out,
        crop_size=crop_size,
        device=device,
    )

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--pth_in", required=True)
    p.add_argument("--ann_root", required=True)
    p.add_argument("--image_root", required=True)

    p.add_argument("--pth_out", default=None)
    p.add_argument("--onnx_out", default=None)

    p.add_argument("--cache_root", default="data/processed")
    p.add_argument("--model_module", default="src.models.test")

    p.add_argument("--amount", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--crop_size", type=int, default=112)
    p.add_argument("--aug_cfg", default="none")

    p.add_argument("--device", default="auto")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    pth_out, onnx_out = make_default_outputs(args.pth_in, args.amount)

    if args.pth_out is not None:
        pth_out = Path(args.pth_out)

    if args.onnx_out is not None:
        onnx_out = Path(args.onnx_out)

    print(f"[compress] pth_in   = {args.pth_in}")
    print(f"[compress] pth_out  = {pth_out}")
    print(f"[compress] onnx_out = {onnx_out}")
    print(f"[compress] device   = {device}")

    model_builder = load_model_builder(args.model_module)

    train_loader, val_loader = build_loaders(
        ann_root=args.ann_root,
        image_root=args.image_root,
        cache_root=args.cache_root,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        aug_cfg=args.aug_cfg,
    )

    compress_from_pth(
        pth_path=args.pth_in,
        pth_out=pth_out,
        onnx_out=onnx_out,
        model_builder=model_builder,
        train_loader=train_loader,
        val_loader=val_loader,
        amount=args.amount,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )


if __name__ == "__main__":
    main()