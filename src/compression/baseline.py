#!/usr/bin/env python3
"""Han et al. Deep Compression baseline.

Full pipeline (the "real thing"; compress.py stays as the pruning-only fallback):

    .pth checkpoint
    -> global magnitude unstructured pruning
    -> fine-tune retraining
    -> per-layer 1-D k-means weight sharing   (conv 8-bit / fc 5-bit, linear init)
    -> centroid fine-tuning                    (fixed index, grouped-gradient update)
    -> Huffman coding of centroid indices
    -> model/gesture_model.ptmodel

The .ptmodel decoder uses ONLY numpy + stdlib (pickle/heapq) so a fresh Colab
inference runtime needs nothing beyond torch + numpy (+ cv2, already pulled in by
predictor.py). No scipy, no sklearn.

Public API:
    compress_from_pth(...)   -> runs the full pipeline, writes .ptmodel
    save_ptmodel(...)        -> encode model -> .ptmodel
    load_ptmodel(path)       -> (model_cfg, state_dict, label_map)   [numpy decode]
    verify_roundtrip(...)    -> assert decode matches the in-memory quantized model
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import itertools
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader




PTMODEL_FORMAT = "ptmodel-dc-v1"
LABEL_NAMES = ["N/A", "fist", "like", "ok", "one", "palm"]


# ── device / builder / data plumbing (self-contained; mirrors compress.py) ────

def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested, but CUDA is not available.")
    return device


def load_model_builder(module_name: str) -> Callable[[dict], nn.Module]:
    module = importlib.import_module(module_name)
    if not hasattr(module, "build_model"):
        raise AttributeError(f"{module_name} must define build_model(model_cfg).")
    return module.build_model


def build_loaders(
    ann_root, image_root, cache_root, crop_size, batch_size, num_workers, device, aug_cfg,
) -> tuple[DataLoader, DataLoader]:
    # Lazy import: keep the .ptmodel decode path (used by inference) free of
    # training-only deps (albumentations, etc.) so a fresh Colab needs only
    # torch + numpy + cv2.
    from src.dataset import HaGRIDv2Dataset
    try:
        from src.augmentation import build_augmentation
    except ImportError:
        build_augmentation = None
    train_transform = None
    if aug_cfg and str(aug_cfg).lower() not in {"none", "null", "false", ""}:
        if build_augmentation is None:
            raise ImportError(
                "Cannot import src.augmentation.build_augmentation. "
                "Install training deps or pass --aug_cfg none."
            )
        train_transform = build_augmentation(aug_cfg, train=True)

    train_ds = HaGRIDv2Dataset(
        ann_root=ann_root, image_root=image_root, cache_root=cache_root,
        split="train", transform=train_transform, crop_size=crop_size,
    )
    val_ds = HaGRIDv2Dataset(
        ann_root=ann_root, image_root=image_root, cache_root=cache_root,
        split="val", transform=None, crop_size=crop_size,
    )

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader


def collect_prunable_params(model: nn.Module) -> list[tuple[nn.Module, str]]:
    params = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    if not params:
        raise RuntimeError("No Conv2d or Linear weights found to prune/quantize.")
    return params


@torch.no_grad()
def quick_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for crop, landmarks, label in loader:
        logits = model(crop.to(device), landmarks.to(device))
        correct += (logits.argmax(dim=1) == label.to(device)).sum().item()
        total += label.numel()
    return correct / max(total, 1)


@torch.no_grad()
def global_sparsity(model: nn.Module) -> float:
    zero = total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach()
            zero += (w == 0).sum().item()
            total += w.numel()
    return zero / max(total, 1)


def prune_and_retrain(
    model, train_loader, val_loader, amount, epochs, lr, device,
) -> nn.Module:
    model = model.to(device)
    params_to_prune = collect_prunable_params(model)

    prune.global_unstructured(
        params_to_prune, pruning_method=prune.L1Unstructured, amount=amount,
    )
    print(f"[baseline] pruning amount={amount:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for crop, landmarks, label in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(crop.to(device), landmarks.to(device)), label.to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        msg = f"[baseline] prune-retrain {epoch + 1}/{epochs} loss={running / max(len(train_loader), 1):.4f}"
        if val_loader is not None:
            msg += f" val_acc={quick_acc(model, val_loader, device):.4f}"
        print(msg)

    for module, name in params_to_prune:
        prune.remove(module, name)  # bake zeros into the dense tensor
    print(f"[baseline] post-prune global sparsity={global_sparsity(model):.4f}")
    return model


# ── 1-D k-means weight sharing ────────────────────────────────────────────────

def kmeans_1d(values: np.ndarray, k: int, iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Linear-init Lloyd on scalar weights. Returns (sorted centroids, index)."""
    values = values.astype(np.float64)
    vmin, vmax = float(values.min()), float(values.max())

    if k <= 1 or vmin == vmax:
        centroids = np.array([values.mean()], dtype=np.float64)
        return centroids, np.zeros(len(values), dtype=np.int64)

    centroids = np.linspace(vmin, vmax, k)  # paper: linear init beats density/random
    index = np.zeros(len(values), dtype=np.int64)

    for _ in range(iters):
        mids = (centroids[:-1] + centroids[1:]) / 2.0
        index = np.searchsorted(mids, values)            # O(n log k), memory-light
        sums = np.bincount(index, weights=values, minlength=k)
        counts = np.bincount(index, minlength=k)
        new = np.where(counts > 0, sums / np.maximum(counts, 1), centroids)
        new.sort()                                       # keep ascending for searchsorted
        if np.allclose(new, centroids):
            centroids = new
            break
        centroids = new

    mids = (centroids[:-1] + centroids[1:]) / 2.0
    index = np.searchsorted(mids, values)
    return centroids, index.astype(np.int64)


def quantize_flat(flat: np.ndarray, mask: np.ndarray, bits: int):
    """Cluster the nonzero entries of `flat`. Returns (centroids f32, index i64, k_eff)."""
    nz = flat[mask]
    k = 1 << bits
    k_eff = int(min(k, np.unique(nz).size))
    centroids, index = kmeans_1d(nz, k_eff)
    return centroids.astype(np.float32), index.astype(np.int64), k_eff


# ── canonical Huffman (numpy + stdlib only) ──────────────────────────────────

def huffman_lengths(freq: dict[int, int]) -> dict[int, int]:
    if len(freq) == 1:
        return {next(iter(freq)): 1}
    cnt = itertools.count()
    heap = [(f, next(cnt), s) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, next(cnt), (n1, n2)))
    root = heap[0][2]

    lengths: dict[int, int] = {}

    def walk(node, depth):
        if isinstance(node, tuple):
            walk(node[0], depth + 1)
            walk(node[1], depth + 1)
        else:
            lengths[node] = depth

    walk(root, 0)
    return lengths


def canonical_codes(code_lengths: np.ndarray) -> dict[int, str]:
    """code_lengths[symbol] -> bit length (0 = absent). Returns symbol -> bitstring."""
    items = sorted((int(l), int(s)) for s, l in enumerate(code_lengths) if l > 0)
    codes: dict[int, str] = {}
    code = 0
    prev_len = items[0][0]
    for i, (length, sym) in enumerate(items):
        if i > 0:
            code = (code + 1) << (length - prev_len)
        codes[sym] = format(code, f"0{length}b")
        prev_len = length
    return codes


def huffman_encode(index: np.ndarray, code_lengths: np.ndarray) -> tuple[bytes, int]:
    codes = canonical_codes(code_lengths)
    codes_int = {s: (int(cs, 2), len(cs)) for s, cs in codes.items()}
    out = bytearray()
    acc = 0
    nbits = 0
    total = 0
    for v in index.tolist():
        val, length = codes_int[v]
        acc = (acc << length) | val
        nbits += length
        total += length
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
    if nbits > 0:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out), total


def huffman_decode(data: bytes, total_bits: int, code_lengths: np.ndarray) -> np.ndarray:
    codes = canonical_codes(code_lengths)
    lookup = {cs: s for s, cs in codes.items()}
    out: list[int] = []
    cur = ""
    read = 0
    for byte in data:
        if read >= total_bits:
            break
        for shift in range(7, -1, -1):
            cur += "1" if (byte >> shift) & 1 else "0"
            read += 1
            sym = lookup.get(cur)
            if sym is not None:
                out.append(sym)
                cur = ""
            if read >= total_bits:
                break
    return np.array(out, dtype=np.int64)

def relindex_encode(mask: np.ndarray, span_bits: int):
    """Han et al. sparse index: gaps between nonzeros (filler-padded) + Huffman.

    Symbol alphabet is [0, 2^span_bits - 1]; the top value is a reserved filler
    meaning "skip that many positions, no nonzero yet". Real nonzero gaps use
    [0, 2^span_bits - 2].
    """
    max_sym = (1 << span_bits) - 1            # filler sentinel
    positions = np.flatnonzero(mask)

    symbols: list[int] = []
    prev = -1
    for p in positions.tolist():
        gap = p - prev - 1
        while gap > max_sym - 1:
            symbols.append(max_sym)           # filler: consume max_sym positions
            gap -= max_sym
        symbols.append(gap)                   # real nonzero, gap in [0, max_sym-1]
        prev = p
    symbols = np.asarray(symbols, dtype=np.int64)

    K = 1 << span_bits
    counts = np.bincount(symbols, minlength=K)
    freq = {s: int(c) for s, c in enumerate(counts) if c > 0}
    code_lengths = np.zeros(K, dtype=np.uint8)
    for s, ln in huffman_lengths(freq).items():
        code_lengths[s] = ln
    bitstream, nbits = huffman_encode(symbols, code_lengths)
    return code_lengths, bitstream, nbits


def relindex_decode(code_lengths, bitstream, nbits, n_weights, span_bits) -> np.ndarray:
    max_sym = (1 << span_bits) - 1
    symbols = huffman_decode(bitstream, nbits, code_lengths)
    mask = np.zeros(n_weights, dtype=bool)
    pos = -1
    for sym in symbols.tolist():
        if sym == max_sym:
            pos += max_sym
        else:
            pos += sym + 1
            mask[pos] = True
    return mask


# ── compression state: quantize / reconstruct / centroid fine-tune ───────────

class DeepCompressionState:
    """Holds per-layer (mask, fixed index, learnable centroids)."""

    def __init__(self) -> None:
        self.layers: dict[nn.Module, dict] = {}

    def quantize(self, prunable: list[tuple[nn.Module, str]], conv_bits: int, fc_bits: int) -> None:
        for module, pname in prunable:
            w = getattr(module, pname).detach().cpu().numpy()
            flat = w.flatten()
            mask = flat != 0.0
            bits = conv_bits if isinstance(module, nn.Conv2d) else fc_bits
            centroids, index, k_eff = quantize_flat(flat, mask, bits)
            self.layers[module] = {
                "pname": pname, "shape": w.shape, "bits": bits, "k": k_eff,
                "mask": mask, "index": index, "centroids": centroids,
            }
            print(f"[baseline] quantized {module.__class__.__name__:<8} "
                  f"shape={tuple(w.shape)} nz={int(mask.sum())} bits={bits} k={k_eff}")

    def _weight_tensor(self, L: dict, device: str) -> torch.Tensor:
        flat = np.zeros(L["mask"].shape[0], dtype=np.float32)
        flat[L["mask"]] = L["centroids"][L["index"]]
        return torch.from_numpy(flat.reshape(L["shape"]).copy()).to(device)

    def reconstruct_all(self, device: str) -> None:
        for module, L in self.layers.items():
            getattr(module, L["pname"]).data = self._weight_tensor(L, device)

    def centroid_step(self, lr: float) -> None:
        """Han et al.: sum gradients per cluster, descend on the centroid value."""
        for module, L in self.layers.items():
            grad = getattr(module, L["pname"]).grad
            if grad is None:
                continue
            g = grad.detach().cpu().numpy().flatten()[L["mask"]]
            cgrad = np.bincount(L["index"], weights=g, minlength=L["k"])
            L["centroids"] = (L["centroids"] - lr * cgrad).astype(np.float32)


def centroid_finetune(model, state, loader, epochs, lr, device) -> None:
    if epochs <= 0:
        state.reconstruct_all(device)
        return
    model.eval()  # freeze BN running stats / dropout; only centroids move
    for p in model.parameters():
        p.requires_grad_(False)
    for module, L in state.layers.items():
        getattr(module, L["pname"]).requires_grad_(True)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        state.reconstruct_all(device)
        running = 0.0
        for crop, landmarks, label in loader:
            for module, L in state.layers.items():
                getattr(module, L["pname"]).grad = None
            loss = criterion(model(crop.to(device), landmarks.to(device)), label.to(device))
            loss.backward()
            state.centroid_step(lr)
            state.reconstruct_all(device)  # apply updated centroids before next batch
            running += loss.item()
        print(f"[baseline] centroid-ft {epoch + 1}/{epochs} "
              f"loss={running / max(len(loader), 1):.4f}")

    for p in model.parameters():
        p.requires_grad_(True)
    state.reconstruct_all(device)


# ── .ptmodel container: encode / decode ──────────────────────────────────────

def save_ptmodel(model, state, model_cfg, path, meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    name_of = {m: n for n, m in model.named_modules()}
    sd = model.state_dict()
    tensors: dict[str, dict] = {}
    quantized: set[str] = set()

    for module, L in state.layers.items():
        full = f"{name_of[module]}.{L['pname']}".lstrip(".")
        quantized.add(full)

        # value index: k-means centroids + Huffman
        counts = np.bincount(L["index"], minlength=L["k"])
        freq = {s: int(c) for s, c in enumerate(counts) if c > 0}
        code_lengths = np.zeros(L["k"], dtype=np.uint8)
        for s, ln in huffman_lengths(freq).items():
            code_lengths[s] = ln
        bitstream, nbits = huffman_encode(L["index"], code_lengths)

        # sparse structure: pick the smaller of {bitmask, relindex+Huffman}
        mask = L["mask"]
        n_w = int(mask.shape[0])
        n_nz = int(mask.sum())
        mask_packed = np.packbits(mask.astype(np.uint8))
        sparse = {"sparse_enc": "bitmask", "mask_packed": mask_packed}

        if 0 < n_nz < n_w:
            rcl, rbs, rnb = relindex_encode(mask, span_bits=L["bits"])
            if int(rcl.nbytes) + len(rbs) < int(mask_packed.nbytes):
                sparse = {
                    "sparse_enc": "relindex",
                    "span_bits": int(L["bits"]),
                    "rel_code_lengths": rcl,
                    "rel_bitstream": rbs,
                    "rel_nbits": int(rnb),
                }

        entry = {
            "kind": "q",
            "shape": list(L["shape"]),
            "bits": int(L["bits"]),
            "k": int(L["k"]),
            "n_weights": n_w,
            "centroids": L["centroids"].astype(np.float32),
            "code_lengths": code_lengths,
            "bitstream": bitstream,
            "nbits": int(nbits),
        }
        entry.update(sparse)
        tensors[full] = entry
        print(f"[baseline] encode {full:<28} sparse={entry['sparse_enc']} nz={n_nz}/{n_w}")

    for name, t in sd.items():
        if name in quantized:
            continue
        tensors[name] = {"kind": "raw", "array": t.detach().cpu().numpy()}

    blob = {
        "format": PTMODEL_FORMAT,
        "model_cfg": model_cfg,
        "label_map": LABEL_NAMES,
        "compression": meta,
        "tensors": tensors,
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f, protocol=4)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[baseline] saved .ptmodel -> {path} ({size_mb:.4f} MB)")


def load_ptmodel(path):
    """numpy/stdlib decode -> (model_cfg, state_dict, label_map, meta)."""
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if blob.get("format") != PTMODEL_FORMAT:
        raise ValueError(f"Unexpected .ptmodel format: {blob.get('format')}")

    state_dict = {}
    for name, t in blob["tensors"].items():
        if t["kind"] == "raw":
            state_dict[name] = torch.from_numpy(np.asarray(t["array"]).copy())
            continue

        n_w = t["n_weights"]
        if t.get("sparse_enc", "bitmask") == "relindex":
            mask = relindex_decode(t["rel_code_lengths"], t["rel_bitstream"],
                                   t["rel_nbits"], n_w, t["span_bits"])
        else:
            mask = np.unpackbits(t["mask_packed"])[:n_w].astype(bool)

        index = huffman_decode(t["bitstream"], t["nbits"], t["code_lengths"])
        flat = np.zeros(n_w, dtype=np.float32)
        flat[mask] = t["centroids"][index]
        state_dict[name] = torch.from_numpy(flat.reshape(t["shape"]).copy())

    return blob["model_cfg"], state_dict, blob.get("label_map"), blob.get("compression", {})


def verify_roundtrip(model, ptmodel_path, atol: float = 1e-6) -> None:
    _, decoded, _, _ = load_ptmodel(ptmodel_path)
    ref = model.state_dict()
    bad = [name for name, ref_t in ref.items()
           if name not in decoded
           or not torch.allclose(ref_t.cpu().float(), decoded[name].float(), atol=atol)]
    if bad:
        raise AssertionError(f"[baseline] decode mismatch on: {bad}")
    print(f"[baseline] decode round-trip OK ({len(ref)} tensors match)")


# ── full pipeline ─────────────────────────────────────────────────────────────
def _spec_score(gts: np.ndarray, finals: np.ndarray) -> int:
    """Challenge scoring: target +1 if correct else -2; N/A false trigger -2."""
    tgt = gts != 0
    s = int(((finals[tgt] == gts[tgt]).astype(int) - 2 * (finals[tgt] != gts[tgt])).sum())
    s += int((-2) * (finals[~tgt] != 0).sum())
    return s


@torch.no_grad()
def calibrate_threshold(model, loader, device, grid=None):
    """Sweep conf_threshold on val to maximize the challenge score.

    Mirrors predictor.decide_class: a non-N/A argmax with conf < tau -> N/A.
    (landmark_gate is a no-op for now, so it is omitted here.)
    """
    if loader is None:
        return 0.5, {}

    model.eval()
    gts, preds, confs = [], [], []
    for crop, landmarks, label in loader:
        probs = torch.softmax(model(crop.to(device), landmarks.to(device)), dim=1).cpu().numpy()
        p = probs.argmax(1)
        gts.append(label.numpy())
        preds.append(p)
        confs.append(probs[np.arange(len(p)), p])
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    confs = np.concatenate(confs)

    if grid is None:
        grid = np.round(np.linspace(0.30, 0.95, 66), 4)

    scored = [(t, _spec_score(gts, np.where((preds != 0) & (confs >= t), preds, 0))) for t in grid]
    best = max(s for _, s in scored)
    best_tau = float(np.median([t for t, s in scored if s == best]))  # middle of optimal plateau

    info = {
        "best_conf_threshold": best_tau,
        "calib_raw_score": int(best),
        "calib_max_raw": int((gts != 0).sum()),
        "calib_n_val": int(len(gts)),
    }
    print(f"[baseline] calibrated conf_threshold={best_tau:.3f} "
          f"(raw={int(best)}/{int((gts != 0).sum())} on {len(gts)} val)")
    return best_tau, info

def compress_from_pth(
    pth_in, pruned_pth_out, quant_pth_out, ptmodel_out,
    model_builder, train_loader, val_loader,
    amount, prune_epochs, prune_lr, conv_bits, fc_bits, ft_epochs, ft_lr, device,
) -> nn.Module:
    device = resolve_device(device)

    ckpt = torch.load(pth_in, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain 'model_state_dict'.")
    model_cfg = ckpt.get("model_cfg", {})

    model = model_builder(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    model = prune_and_retrain(model, train_loader, val_loader,
                              amount, prune_epochs, prune_lr, device)
    if pruned_pth_out is not None:
        Path(pruned_pth_out).parent.mkdir(parents=True, exist_ok=True)
        pruned = dict(ckpt); pruned["model_state_dict"] = model.state_dict()
        torch.save(pruned, pruned_pth_out)
        print(f"[baseline] saved pruned PTH -> {pruned_pth_out}")

    state = DeepCompressionState()
    state.quantize(collect_prunable_params(model), conv_bits, fc_bits)

    centroid_finetune(model, state, train_loader, ft_epochs, ft_lr, device)
    if val_loader is not None:
        print(f"[baseline] post-quant val_acc={quick_acc(model, val_loader, device):.4f}")

    if quant_pth_out is not None:
        Path(quant_pth_out).parent.mkdir(parents=True, exist_ok=True)
        q = dict(ckpt); q["model_state_dict"] = model.state_dict()
        q["compression"] = {"conv_bits": conv_bits, "fc_bits": fc_bits,
                            "amount": amount, "sparsity": global_sparsity(model)}
        torch.save(q, quant_pth_out)
        print(f"[baseline] saved quantized PTH -> {quant_pth_out}")

    # calibrate N/A confidence threshold on the quantized model
    best_tau, calib = calibrate_threshold(model, val_loader, device)

    meta = {
        "method": "deep_compression",
        "prune_amount": amount, "conv_bits": conv_bits, "fc_bits": fc_bits,
        "global_sparsity": global_sparsity(model),
        **calib,  # includes best_conf_threshold + calib stats
    }
    save_ptmodel(model, state, model_cfg, ptmodel_out, meta)
    verify_roundtrip(model, ptmodel_out)
    return model

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pth_in", required=True)
    p.add_argument("--ann_root", required=True)
    p.add_argument("--image_root", required=True)

    p.add_argument("--pruned_pth_out", default=None)
    p.add_argument("--quant_pth_out", default=None)
    p.add_argument("--ptmodel_out", default="model/gesture_model.ptmodel")

    p.add_argument("--cache_root", default="data/processed")
    p.add_argument("--model_module", default="src.models.test")

    p.add_argument("--amount", type=float, default=0.5)
    p.add_argument("--prune_epochs", type=int, default=3)
    p.add_argument("--prune_lr", type=float, default=1e-4)

    p.add_argument("--conv_bits", type=int, default=8)   # paper: 8-bit conv
    p.add_argument("--fc_bits", type=int, default=5)     # paper: 5-bit fc
    p.add_argument("--ft_epochs", type=int, default=2)
    p.add_argument("--ft_lr", type=float, default=1e-4)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--crop_size", type=int, default=112)
    p.add_argument("--aug_cfg", default="none")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    pth_in = Path(args.pth_in)
    pruned_out = args.pruned_pth_out
    quant_out = args.quant_pth_out
    if pruned_out is None:
        tag = int(round(args.amount * 100))
        pruned_out = str(pth_in.with_name(f"{pth_in.stem}_pruned{tag}.pth"))
        quant_out = str(pth_in.with_name(f"{pth_in.stem}_pruned{tag}_quant.pth"))

    print(f"[baseline] pth_in     = {pth_in}")
    print(f"[baseline] pruned_out = {pruned_out}")
    print(f"[baseline] quant_out  = {quant_out}")
    print(f"[baseline] ptmodel    = {args.ptmodel_out}")
    print(f"[baseline] device     = {device}")

    model_builder = load_model_builder(args.model_module)
    train_loader, val_loader = build_loaders(
        ann_root=args.ann_root, image_root=args.image_root, cache_root=args.cache_root,
        crop_size=args.crop_size, batch_size=args.batch_size, num_workers=args.num_workers,
        device=device, aug_cfg=args.aug_cfg,
    )

    compress_from_pth(
        pth_in=args.pth_in,
        pruned_pth_out=pruned_out, quant_pth_out=quant_out, ptmodel_out=args.ptmodel_out,
        model_builder=model_builder, train_loader=train_loader, val_loader=val_loader,
        amount=args.amount, prune_epochs=args.prune_epochs, prune_lr=args.prune_lr,
        conv_bits=args.conv_bits, fc_bits=args.fc_bits,
        ft_epochs=args.ft_epochs, ft_lr=args.ft_lr, device=device,
    )


if __name__ == "__main__":
    main()