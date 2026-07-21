
"""CSPDarknet backbone: assembles the blocks into dark3/dark4/dark5. See textbook Ch.4."""
from __future__ import annotations

import torch
import torch.nn as nn

from prism.models.blocks import ConvBNAct, CSPLayer, SPPBottleneck


class Focus(nn.Module):
    """Downsample by slicing, not by throwing pixels away. Take every-other-pixel
    into 4 sub-images, stack them on the channel axis (3 -> 12 channels at half
    resolution), then conv. Halves H and W losslessly. This is YOLOX's stem."""

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = ConvBNAct(in_ch * 4, out_ch, k=k, stride=1)

    def forward(self, x):
        tl = x[..., ::2, ::2]     # top-left pixels
        bl = x[..., 1::2, ::2]    # bottom-left
        tr = x[..., ::2, 1::2]    # top-right
        br = x[..., 1::2, 1::2]   # bottom-right
        return self.conv(torch.cat((tl, bl, tr, br), dim=1))


class CSPDarknet(nn.Module):
    """YOLOX-S backbone. Emits three feature maps: dark3 (stride 8), dark4 (16),
    dark5 (32) -- at 128/256/512 channels for a 640x512 input."""

    def __init__(self, base_ch=32, base_depth=1):
        super().__init__()
        self.stem = Focus(3, base_ch, k=3)                         # -> stride 2

        self.dark2 = nn.Sequential(
            ConvBNAct(base_ch, base_ch * 2, k=3, stride=2),         # -> stride 4
            CSPLayer(base_ch * 2, base_ch * 2, n=base_depth),
        )
        self.dark3 = nn.Sequential(
            ConvBNAct(base_ch * 2, base_ch * 4, k=3, stride=2),     # -> stride 8
            CSPLayer(base_ch * 4, base_ch * 4, n=base_depth * 3),
        )
        self.dark4 = nn.Sequential(
            ConvBNAct(base_ch * 4, base_ch * 8, k=3, stride=2),     # -> stride 16
            CSPLayer(base_ch * 8, base_ch * 8, n=base_depth * 3),
        )
        self.dark5 = nn.Sequential(
            ConvBNAct(base_ch * 8, base_ch * 16, k=3, stride=2),    # -> stride 32
            SPPBottleneck(base_ch * 16, base_ch * 16),
            CSPLayer(base_ch * 16, base_ch * 16, n=base_depth, shortcut=False),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        c3 = self.dark3(x)      # stride 8
        c4 = self.dark4(c3)     # stride 16
        c5 = self.dark5(c4)     # stride 32
        return c3, c4, c5
