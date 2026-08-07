"""CSPDarknet backbone: assembles the blocks into dark3/dark4/dark5. See textbook Ch.4."""

from __future__ import annotations

import torch
from torch import nn

from prism.models.blocks import ConvBNAct, CSPLayer, SPPBottleneck


class Focus(nn.Module):
    """Downsample by slicing, not by throwing pixels away. Take every-other-pixel
    into 4 sub-images, stack them on the channel axis (3 -> 12 channels at half
    resolution), then conv. Halves H and W losslessly. This is YOLOX's stem.

    ``in_ch`` is 3 for the unimodal backbones and 6 for the early-fusion stem,
    which concatenates RGB + thermal BEFORE the first conv (the fusion happens
    at the input, so the single stream never splits)."""

    def __init__(self, in_ch=3, out_ch=32, k=3):
        super().__init__()
        self.conv = ConvBNAct(in_ch * 4, out_ch, k=k, stride=1)

    def forward(self, x):
        tl = x[..., ::2, ::2]  # top-left pixels
        bl = x[..., 1::2, ::2]  # bottom-left
        tr = x[..., ::2, 1::2]  # top-right
        br = x[..., 1::2, 1::2]  # bottom-right
        return self.conv(torch.cat((tl, bl, tr, br), dim=1))


class CSPDarknet(nn.Module):
    """YOLOX-S backbone. Emits three feature maps: dark3 (stride 8), dark4 (16),
    dark5 (32) -- at 128/256/512 channels for a 640x512 input."""

    def __init__(self, base_ch=32, base_depth=1, in_ch=3):
        super().__init__()
        self.stem = Focus(in_ch, base_ch, k=3)  # -> stride 2

        self.dark2 = nn.Sequential(
            ConvBNAct(base_ch, base_ch * 2, k=3, stride=2),  # -> stride 4
            CSPLayer(base_ch * 2, base_ch * 2, n=base_depth),
        )
        self.dark3 = nn.Sequential(
            ConvBNAct(base_ch * 2, base_ch * 4, k=3, stride=2),  # -> stride 8
            CSPLayer(base_ch * 4, base_ch * 4, n=base_depth * 3),
        )
        self.dark4 = nn.Sequential(
            ConvBNAct(base_ch * 4, base_ch * 8, k=3, stride=2),  # -> stride 16
            CSPLayer(base_ch * 8, base_ch * 8, n=base_depth * 3),
        )
        self.dark5 = nn.Sequential(
            ConvBNAct(base_ch * 8, base_ch * 16, k=3, stride=2),  # -> stride 32
            SPPBottleneck(base_ch * 16, base_ch * 16),
            CSPLayer(base_ch * 16, base_ch * 16, n=base_depth, shortcut=False),
        )

    def forward(self, x):
        feats = self.forward_upto(x, 5)
        return feats[3], feats[4], feats[5]

    def forward_upto(self, x, level: int):
        """Compute the backbone only up to ``level`` (2..5).

        Returns {level: feature map} for every level in 2..level. One-shot fusion
        uses this so the stream that meets early never computes (or pays FLOPs
        for) the levels that only exist after the shared stream takes over.
        """
        x = self.stem(x)
        x = self.dark2(x)
        feats = {2: x}
        if level == 2:
            return feats
        x = self.dark3(x)
        feats[3] = x
        if level == 3:
            return feats
        x = self.dark4(x)
        feats[4] = x
        if level == 4:
            return feats
        x = self.dark5(x)
        feats[5] = x
        return feats
