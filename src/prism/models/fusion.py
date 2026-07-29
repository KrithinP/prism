"""Fusion operators: how two feature maps (one per camera) become one.

The first and default operator is concat + 1x1 conv (textbook Ch.10): stack the two
maps channel-wise, then a learned 1x1 convolution mixes them back to the original
channel count. This is the workhorse the field uses and the one every other operator
in the study must beat. Later operators (sum, gating, attention) will live here too;
they all share the signature fuse(rgb_feat, thermal_feat) -> fused_feat of the same
shape as either input.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from prism.models.blocks import ConvBNAct


class ConcatFuse(nn.Module):
    """Concatenate the two feature maps (2*C channels) and project back to C with a
    learned 1x1 conv. The 1x1 learns a per-channel blend of the two cameras."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = ConvBNAct(2 * channels, channels, k=1)

    def forward(self, rgb_feat, thermal_feat):
        return self.proj(torch.cat([rgb_feat, thermal_feat], dim=1))


class SumFuse(nn.Module):
    """Zero-parameter element-wise sum. The honest floor operator, and the default at
    stages that a single-stage fusion config leaves un-fused."""

    def forward(self, rgb_feat, thermal_feat):
        return rgb_feat + thermal_feat


# name -> operator class, so experiments select the operator by string.
OPERATORS = {"concat": ConcatFuse, "sum": SumFuse}
