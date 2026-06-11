#!/usr/bin/env python3
"""run.py — 一個指令跑完 train -> compress -> evaluate。

設計重點：
  1. 全程在同一個 Python process，直接呼叫各模組的 function，不用 subprocess。
  2. 所有產出以 --name 命名，不再寫死 gesture_model，方便區分實驗。
  3. --model_module 從 train 到 compress 全程共用，預設 src.models.test。
     compress 時會把 model_module 寫進 .ptmodel meta，inference / evaluate
     之後自動還原架構，不需要再手動指定。

產出（全部以 <name> 為檔名）：
    <checkpoints_dir>/<name>.pth                       (train)
    <checkpoints_dir>/<name>_pruned<P>.pth             (compress 中繼)
    <checkpoints_dir>/<name>_pruned<P>_quant.pth       (compress 中繼)
    <model_dir>/<name>.ptmodel                         (compress 最終)

其中 <P> = round(amount * 100)，例如 amount=0.5 -> pruned50。

階段控制：
    --skip_train       跳過 train，用現有 <checkpoints_dir>/<name>.pth
    --skip_compress    跳過 compress，用現有 <model_dir>/<name>.ptmodel
    --skip_eval        compress 完就停

範例：
    # 完整跑
    python run.py --name resnet_aug --model_module src.models.resnet \\
        --epochs 10 --mini_train --aug_cfg configs/aug.yaml

    # 已有 .pth，只重跑 compress + eval
    python run.py --name resnet_aug --model_module src.models.resnet \\
        --skip_train --amount 0.6 --prune_epochs 5

    # 已有 .ptmodel，只重跑 eval
    python run.py --name resnet_aug --skip_train --skip_compress --split test

提交說明：
    inference.py 固定讀 model/gesture_model.ptmodel（TA harness 規定）。
    選定實驗後：cp model/<name>.ptmodel model/gesture_model.ptmodel
    架構從 .ptmodel meta 自動還原，不需要改 inference.py。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _require(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"[run] ERROR: expected file not found: {path}", file=sys.stderr)
        print(f"[run] {hint}", file=sys.stderr)
        sys.exit(1)


def _banner(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"[run] {text}")
    print(f"{'=' * 70}", flush=True)


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="train -> compress -> evaluate in one command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # naming
    p.add_argument("--name", required=True,
                   help="experiment name; every output file uses this as stem")
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--model_dir", default="model")

    # model architecture (shared across train + compress)
    p.add_argument("--model_module", default="src.models.test",
                   help="module under src/models/ that defines build_model(model_cfg)")

    # pipeline control
    p.add_argument("--skip_train",    action="store_true")
    p.add_argument("--skip_compress", action="store_true")
    p.add_argument("--skip_eval",     action="store_true")

    # shared
    p.add_argument("--data_root",   default="data")
    p.add_argument("--mini_train",  action="store_true")
    p.add_argument("--crop_size",   type=int,   default=112)
    p.add_argument("--device",      default="auto",
                   help="auto / cuda / cpu")

    # train
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--img_dim",     type=int,   default=128)
    p.add_argument("--lm_dim",      type=int,   default=32)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--aug_cfg",     default="none")

    # compress
    p.add_argument("--amount",               type=float, default=0.5)
    p.add_argument("--prune_epochs",         type=int,   default=3)
    p.add_argument("--prune_lr",             type=float, default=1e-4)
    p.add_argument("--conv_bits",            type=int,   default=8)
    p.add_argument("--fc_bits",              type=int,   default=5)
    p.add_argument("--ft_epochs",            type=int,   default=2)
    p.add_argument("--ft_lr",               type=float, default=1e-4)
    p.add_argument("--compress_batch_size",  type=int,   default=16)
    p.add_argument("--compress_num_workers", type=int,   default=0)

    # evaluate
    p.add_argument("--split",          default="test")
    p.add_argument("--conf_threshold", type=float, default=None,
                   help="override predictor confidence threshold; default uses calibrated value")

    return p.parse_args()


# ── device ────────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ── stage 1: train ────────────────────────────────────────────────────────────

def stage_train(args, pth_out: Path, device: str) -> None:
    _banner("STAGE 1/3  train")

    import importlib
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import train as train_mod
    from src.compression.baseline import calibrate_threshold

    data_root = Path(args.data_root)

    # build paths (mirrors train.py main())
    image_root  = str(data_root / train_mod.IMAGE_SUBDIR)
    ann_root    = str(data_root / train_mod.ANN_SUBDIR)
    cache_root  = str(data_root / train_mod.CACHE_SUBDIR)

    if args.mini_train:
        from src.build_mini_train import build_mini_train
        mini_dir = data_root / train_mod.MINI_SUBDIR
        print("[run] --mini_train: building mini subset...")
        build_mini_train(image_root=image_root, output_dir=mini_dir, per_class=2000)
        ann_root   = str(mini_dir / train_mod.ANN_SUBDIR)
        cache_root = str(mini_dir / train_mod.CACHE_SUBDIR)

    # reuse train.py's DataLoader builder
    import argparse as _ap
    train_args = _ap.Namespace(
        data_root=args.data_root,
        mini_train=args.mini_train,
        image_root=image_root,
        ann_root=ann_root,
        cache_root=cache_root,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_cfg=args.aug_cfg,
        device=device,
    )
    train_loader, val_loader = train_mod.build_loaders(train_args)

    # build model via the specified module
    module = importlib.import_module(args.model_module)
    build_model = module.build_model

    model_cfg = {
        "crop_size": args.crop_size,
        "img_dim":   args.img_dim,
        "lm_dim":    args.lm_dim,
    }
    model = build_model(model_cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_tau = 0.5

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for crop, lm, label in pbar:
            crop, lm, label = crop.to(device), lm.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(crop, lm), label)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_acc = train_mod.evaluate_acc(model, val_loader, device)
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"loss={running/max(len(train_loader),1):.4f}  val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_tau, _ = calibrate_threshold(model, val_loader, device)

    # save checkpoint
    pth_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "best_conf_threshold": best_tau,
        "model_state_dict": model.state_dict(),
        "model_cfg": model_cfg,
        "label_map": train_mod.LABEL_MAP,
        "val_acc": best_acc,
        "aug_cfg": None if str(args.aug_cfg).lower() in {"", "none", "null", "false"} else args.aug_cfg,
    }, pth_out)
    print(f"[run] saved checkpoint -> {pth_out}  (best val_acc={best_acc:.4f})")


# ── stage 2: compress ─────────────────────────────────────────────────────────

def stage_compress(args, pth_in: Path, pruned_out: Path, quant_out: Path,
                   ptmodel_out: Path, device: str) -> None:
    _banner("STAGE 2/3  compress")

    import importlib
    from src.compression.baseline import (
        compress_from_pth, build_loaders as compress_build_loaders,
        resolve_device,
    )

    device = resolve_device(device)
    data_root = Path(args.data_root)
    image_root = str(data_root / "hagridv2_512")

    if args.mini_train:
        mini_dir   = data_root / "mini_train"
        ann_root   = str(mini_dir / "annotations")
        cache_root = str(mini_dir / "processed")
    else:
        ann_root   = str(data_root / "annotations")
        cache_root = str(data_root / "processed")

    module = importlib.import_module(args.model_module)
    model_builder = module.build_model

    train_loader, val_loader = compress_build_loaders(
        ann_root=ann_root, image_root=image_root, cache_root=cache_root,
        crop_size=args.crop_size, batch_size=args.compress_batch_size,
        num_workers=args.compress_num_workers, device=device,
        aug_cfg=args.aug_cfg,
    )

    ptmodel_out.parent.mkdir(parents=True, exist_ok=True)
    compress_from_pth(
        pth_in=pth_in,
        pruned_pth_out=pruned_out,
        quant_pth_out=quant_out,
        ptmodel_out=ptmodel_out,
        model_builder=model_builder,
        train_loader=train_loader,
        val_loader=val_loader,
        amount=args.amount,
        prune_epochs=args.prune_epochs,
        prune_lr=args.prune_lr,
        conv_bits=args.conv_bits,
        fc_bits=args.fc_bits,
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        device=device,
        model_module=args.model_module,
    )
    print(f"[run] compress output -> {ptmodel_out}")


# ── stage 3: evaluate ─────────────────────────────────────────────────────────

def stage_eval(args, ptmodel_out: Path, device: str) -> None:
    _banner("STAGE 3/3  evaluate")

    from src.evaluate import run_evaluate
    run_evaluate(
        weights_path=ptmodel_out,
        data_root=args.data_root,
        mini_train=args.mini_train,
        split=args.split,
        crop_size=args.crop_size,
        conf_threshold=args.conf_threshold,
        device=device,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    ckpt_dir  = Path(args.checkpoints_dir)
    model_dir = Path(args.model_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    tag          = int(round(args.amount * 100))
    pth_path     = ckpt_dir / f"{args.name}.pth"
    pruned_path  = ckpt_dir / f"{args.name}_pruned{tag}.pth"
    quant_path   = ckpt_dir / f"{args.name}_pruned{tag}_quant.pth"
    ptmodel_path = model_dir / f"{args.name}.ptmodel"

    stages = [
        ("train",    not args.skip_train),
        ("compress", not args.skip_compress),
        ("eval",     not args.skip_eval),
    ]
    active = [s for s, on in stages if on]

    print(f"[run] name         = {args.name}")
    print(f"[run] model_module = {args.model_module}")
    print(f"[run] device       = {device}")
    print(f"[run] pth          = {pth_path}")
    print(f"[run] ptmodel      = {ptmodel_path}")
    print(f"[run] stages       = {' -> '.join(active) if active else '(nothing)'}")

    if not args.skip_train:
        stage_train(args, pth_path, device)
    else:
        print("\n[run] --skip_train: using existing checkpoint")
        _require(pth_path, f"run without --skip_train, or provide {pth_path}")

    if not args.skip_compress:
        _require(pth_path, f"need {pth_path} before compress")
        stage_compress(args, pth_path, pruned_path, quant_path, ptmodel_path, device)
    else:
        print("\n[run] --skip_compress: using existing .ptmodel")
        if not args.skip_eval:
            _require(ptmodel_path, f"run without --skip_compress, or provide {ptmodel_path}")

    if not args.skip_eval:
        stage_eval(args, ptmodel_path, device)

    print("\n[run] pipeline finished.")
    print(f"[run] checkpoint : {pth_path}")
    print(f"[run] ptmodel    : {ptmodel_path}")
    print(f"[run] to submit  : cp {ptmodel_path} {model_dir / 'gesture_model.ptmodel'}")


if __name__ == "__main__":
    main()