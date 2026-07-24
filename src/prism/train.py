
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

from prism.data.kaist import PairedKAIST, kaist_split
from prism.data.detection import DetectionDataset, detection_collate
from prism.data.transforms import Compose, PairedHFlip, RGBPhotometric
from prism.models.yolox import YOLOX
from prism.models.loss import YOLOXLoss


class Diverged(RuntimeError):
    """Training blew up at this learning rate. A finding, not a bug."""


# Healthy loss on this task sits in the single digits; hundreds means it is running away.
DIVERGENCE_LOSS_LIMIT = 500.0


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


def require_device(allow_cpu=False):
    """Refuse to train on CPU unless explicitly allowed.

    A silent CPU fallback is worse than a crash: a full run is ~100x slower and looks
    like the model is merely slow to converge. This bit us on 2026-07-24 when an NVIDIA
    driver upgrade left the loaded kernel module (580.159.03) out of sync with the
    userspace libs (580.173.02) -- nvidia-smi reported "Driver/library version mismatch",
    torch.cuda.is_available() went False, and a sweep started grinding on CPU. Fix was a
    reboot. Fail loudly instead.
    """
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        print("WARNING: training on CPU (--allow-cpu). This will be very slow.")
        return "cpu"
    raise RuntimeError(
        "CUDA is not available -- refusing to train on CPU.\n"
        "  Check:  nvidia-smi   (a 'Driver/library version mismatch' means the loaded\n"
        "          kernel module and the installed userspace driver disagree -- reboot)\n"
        "  Then:   python -c \"import torch; print(torch.cuda.is_available())\"\n"
        "  Override with --allow-cpu if you really mean it."
    )


def train(root, modality="rgb", lr=0.01, epochs=10, batch_size=8,
          seed=0, limit=None, out="checkpoints", positives_only=False,
          split="train", allow_cpu=False):
    set_seed(seed)
    device = require_device(allow_cpu)
    # Create the checkpoint dir UP FRONT: a bad path must fail in a second, not after
    # an hour of training (it did exactly that on 2026-07-24 and cost a full 8-epoch run).
    Path(out).mkdir(parents=True, exist_ok=True)

    aug = Compose([PairedHFlip(0.5), RGBPhotometric(0.2, 0.2)])
    # Official protocol split: sets 00-05, every 2nd frame, sanitized annotations.
    # Real runs use ALL train frames -- background-only frames are fine (and needed)
    # now that the prior-bias init keeps the from-scratch objectness loss sane.
    base = kaist_split(root, split)
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

            # A diverged run is a RESULT ("this lr is unstable"), not a crash. Detect it
            # here and stop cleanly, before NaNs propagate into kernels that abort the
            # process. Seen 2026-07-24: lr=0.02 ran obj 12 -> 989 and took down the whole
            # sweep via a device-side assert.
            if not torch.isfinite(loss):
                raise Diverged(f"non-finite loss at step {step} (lr={lr})")
            if loss.item() > DIVERGENCE_LOSS_LIMIT:
                raise Diverged(f"loss {loss.item():.1f} exceeded "
                               f"{DIVERGENCE_LOSS_LIMIT} at step {step} (lr={lr})")

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

    Path(out).mkdir(parents=True, exist_ok=True)   # parents: sweep uses nested dirs
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
    ap.add_argument("--split", default="train",
                    choices=["train", "test", "test-day", "test-night"])
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU training (very slow); default is to fail loudly")
    a = ap.parse_args()
    train(a.root, a.modality, a.lr, a.epochs, a.batch_size, a.seed, a.limit, a.out,
          a.positives_only, a.split, a.allow_cpu)


if __name__ == "__main__":
    main()
