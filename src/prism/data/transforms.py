"""Paired augmentations. Geometric ops hit both modalities + boxes identically;
photometric ops hit RGB only. See textbook Ch.2."""

from __future__ import annotations

import random

import torch


class PairedHFlip:
    """Horizontal flip applied identically to RGB, thermal, and boxes."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, rgb, thermal, boxes):
        if random.random() >= self.p:  # one shared coin flip for BOTH streams
            return rgb, thermal, boxes
        W = rgb.shape[2]  # image width (tensors are C, H, W)
        rgb = torch.flip(rgb, dims=[2])  # flip width axis
        thermal = torch.flip(thermal, dims=[2])  # SAME flip, same axis
        if boxes.numel() > 0:
            # a box [x1, y1, x2, y2] mirrors to [W-x2, y1, W-x1, y2]
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes = boxes.clone()
            boxes[:, 0] = W - x2
            boxes[:, 2] = W - x1
        return rgb, thermal, boxes


class RGBPhotometric:
    """Brightness/contrast jitter on the RGB stream ONLY. Thermal is left untouched
    because its pixel values are a physical measurement, not a look."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, rgb, thermal, boxes):
        rgb = rgb.float()
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = rgb.mean()
        rgb = (rgb - mean) * c + mean * b  # contrast about the mean, then brightness
        rgb = rgb.clamp(0, 255)
        return rgb, thermal, boxes  # thermal returned unchanged, on purpose


def translate_with_zeros(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate a CHW tensor without wraparound; positive dx/dy move right/down."""
    _, height, width = image.shape
    result = torch.zeros_like(image)
    src_x1, src_x2 = max(0, -dx), min(width, width - dx)
    src_y1, src_y2 = max(0, -dy), min(height, height - dy)
    dst_x1, dst_x2 = max(0, dx), min(width, width + dx)
    dst_y1, dst_y2 = max(0, dy), min(height, height + dy)
    if src_x1 < src_x2 and src_y1 < src_y2:
        result[:, dst_y1:dst_y2, dst_x1:dst_x2] = image[:, src_y1:src_y2, src_x1:src_x2]
    return result


class PairedModalityDropout:
    """Drop exactly one complete modality with probability ``p``.

    RGB and thermal are selected with equal probability. This is a robustness
    intervention for dedicated Block D runs, not part of the core architecture
    comparison recipe.
    """

    def __init__(self, p: float = 0.2):
        if not 0 <= p <= 1:
            raise ValueError("modality-dropout probability must be in [0, 1]")
        self.p = p

    def __call__(self, rgb, thermal, boxes):
        if random.random() < self.p:
            if random.random() < 0.5:
                rgb = torch.zeros_like(rgb)
            else:
                thermal = torch.zeros_like(thermal)
        return rgb, thermal, boxes


class ThermalMisalignment:
    """Randomly translate thermal content while keeping RGB-frame targets fixed."""

    def __init__(self, max_pixels: int = 8):
        if max_pixels < 0:
            raise ValueError("max_pixels must be non-negative")
        self.max_pixels = max_pixels

    def __call__(self, rgb, thermal, boxes):
        if self.max_pixels:
            dx = random.randint(-self.max_pixels, self.max_pixels)
            dy = random.randint(-self.max_pixels, self.max_pixels)
            thermal = translate_with_zeros(thermal, dx, dy)
        return rgb, thermal, boxes


class Compose:
    """Chain transforms; each receives and returns (rgb, thermal, boxes)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, thermal, boxes):
        for t in self.transforms:
            rgb, thermal, boxes = t(rgb, thermal, boxes)
        return rgb, thermal, boxes
