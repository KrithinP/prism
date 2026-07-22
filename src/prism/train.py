
"""From-scratch training loop for a single-modality YOLOX baseline on KAIST.

Deliberately minimal and readable (textbook Ch.7): SGD + linear warmup -> cosine
decay, AMP for the 12GB Blackwell card, per-seed determinism for the 5-seed
protocol, and a checkpoint at the end. The LR is a CLI knob -- this is the loop
Soham's LR sweep drives.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prism.data.kaist import PairedKAIST
from prism.data.detection import DetectionDataset, detection_collate
from prism.data.transforms import Compose, PairedHFlip, RGBPhotometric
from prism.models.yolox import YOLOX
from prism.models.loss import YOLOXLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, lr, momentum=0.9, weight_decay=5e-4):
    """YOLOX-style: no weight decay on BN weights or biases (stabilises from-scratch)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):   # BN scale/shift and all biases
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


def lr_factor(step, total_steps, warmup_steps):
    """Linear warmup 0->1, then cosine 1->0 across the rest of training."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def positive_indices(base):
    """Frame indices with >=1 scored pedestrian (scans XML only, no image decode)."""
    from prism.data.kaist import parse_annotation
    keep = []
    for i, fid in enumerate(base.ids):
        _, _, xml = base._paths(fid)
        if (parse_annotation(xml)["classes"] == 1).any():
            keep.append(i)
    return keep


def train(root, modality="rgb", lr=0.01, epochs=10, batch_size=8,
          seed=0, limit=None, out="checkpoints", positives_only=False):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aug = Compose([PairedHFlip(0.5), RGBPhotometric(0.2, 0.2)])
    base = PairedKAIST(root)
    ds = DetectionDataset(base, modality=modality, transforms=aug)
    idx = positive_indices(base) if positives_only else list(range(len(ds)))
    if limit:                                    # smoke mode: overfit a few hundred frames
        idx = idx[:limit]
    ds = torch.utils.data.Subset(ds, idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4,
                        collate_fn=detection_collate, drop_last=True)

    model = YOLOX(num_classes=1).to(device).train()
    lossfn = YOLOXLoss(num_classes=1)
    opt = build_optimizer(model, lr)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    total_steps = epochs * len(loader)
    warmup_steps = min(500, max(1, total_steps // 20))   # ~5% warmup
    step = 0
    for epoch in range(epochs):
        for imgs, targets in loader:
            imgs = imgs.to(device)
            for g in opt.param_groups:            # apply the schedule this step
                g["lr"] = lr * lr_factor(step, total_steps, warmup_steps)

            opt.zero_grad()
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                loss, parts = lossfn(model(imgs), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                  # undo AMP scaling before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 20 == 0:
                print(f"e{epoch} s{step}/{total_steps} "
                      f"lr{opt.param_groups[0]['lr']:.4f} loss{loss.item():.3f} "
                      f"iou{parts['iou']:.3f} obj{parts['obj']:.3f} "
                      f"cls{parts['cls']:.3f} fg{parts['num_fg']}")

    Path(out).mkdir(exist_ok=True)
    ckpt = Path(out) / f"{modality}_seed{seed}.pt"
    torch.save({"model": model.state_dict(), "modality": modality, "seed": seed}, ckpt)
    print(f"saved {ckpt}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--modality", default="rgb", choices=["rgb", "thermal"])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="smoke: train on first N frames")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--positives-only", action="store_true",
                    help="restrict to frames containing >=1 pedestrian (for the smoke)")
    a = ap.parse_args()
    train(a.root, a.modality, a.lr, a.epochs, a.batch_size, a.seed, a.limit, a.out,
          a.positives_only)


if __name__ == "__main__":
    main()
