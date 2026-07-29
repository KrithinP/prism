
"""Training-ready wrapper over PairedKAIST.

Turns the paired loader's per-frame dict into exactly what the training step needs:
a single-modality image tensor in [0,1] and a (Ni, 5) target [cls, x1, y1, x2, y2].
KAIST frames are all 640x512 (divisible by 32), so no letterbox is needed here.
Ignore regions are dropped from targets for the baseline (see textbook Ch.7 note).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from prism.data.kaist import PairedKAIST


class DetectionDataset(Dataset):
    def __init__(self, base: PairedKAIST, modality: str = "rgb", transforms=None):
        assert modality in ("rgb", "thermal")
        self.base = base
        self.modality = modality
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        s = self.base[i]
        rgb, thermal, boxes = s["rgb"].float(), s["thermal"].float(), s["boxes"]

        # keep only scored positives (person); ignore regions dropped for the baseline
        pos = s["classes"] == 1
        boxes = boxes[pos]

        if self.transforms is not None:
            rgb, thermal, boxes = self.transforms(rgb, thermal, boxes)

        img = (rgb if self.modality == "rgb" else thermal) / 255.0  # -> [0,1]

        # target rows [cls, x1, y1, x2, y2]; model has one class, so cls = 0
        if boxes.numel() > 0:
            cls = torch.zeros(boxes.shape[0], 1)
            target = torch.cat([cls, boxes], dim=1)
        else:
            target = torch.zeros(0, 5)
        return img, target


def detection_collate(batch):
    """Stack images (all same size); keep targets as a list (ragged box counts)."""
    imgs = torch.stack([img for img, _ in batch], dim=0)   # (B, C, H, W)
    targets = [t for _, t in batch]                        # length-B list of (Ni, 5)
    return imgs, targets


class FusionDataset(Dataset):
    """Like DetectionDataset but returns BOTH cameras for the same frame, for the
    two-stream model. The same paired augmentation hits rgb, thermal, and boxes
    together, so the two streams stay in correspondence."""

    def __init__(self, base: PairedKAIST, transforms=None):
        self.base = base
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        s = self.base[i]
        rgb, thermal, boxes = s["rgb"].float(), s["thermal"].float(), s["boxes"]
        boxes = boxes[s["classes"] == 1]                   # scored positives only
        if self.transforms is not None:
            rgb, thermal, boxes = self.transforms(rgb, thermal, boxes)
        rgb, thermal = rgb / 255.0, thermal / 255.0        # -> [0,1]
        if boxes.numel() > 0:
            target = torch.cat([torch.zeros(boxes.shape[0], 1), boxes], dim=1)
        else:
            target = torch.zeros(0, 5)
        return rgb, thermal, target


def fusion_collate(batch):
    """Stack both camera streams; keep targets as a ragged list."""
    rgb = torch.stack([r for r, _, _ in batch], dim=0)
    thermal = torch.stack([t for _, t, _ in batch], dim=0)
    targets = [tg for _, _, tg in batch]
    return rgb, thermal, targets
