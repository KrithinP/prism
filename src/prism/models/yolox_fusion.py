"""Two-stream YOLOX: one backbone per camera, fused, then the SAME neck + head.

Everything after fusion is identical to the single-stream YOLOX (Ch.4) -- deliberately,
so any result difference comes only from the fusion, never from a changed neck/head/loss.
See textbook Ch.10.

The default config is F-cat-345: concat+1x1 fusion at all three backbone scales
(dark3/dark4/dark5). fuse_stages restricts which scales fuse (Phase 2); a stage left
un-fused falls back to a zero-parameter sum so the neck still receives all three scales.
"""
from __future__ import annotations

import torch.nn as nn

from prism.models.darknet import CSPDarknet
from prism.models.pafpn import PAFPN
from prism.models.head import DecoupledHead
from prism.models.fusion import OPERATORS, SumFuse

_STAGES = (3, 4, 5)


class TwoStreamYOLOX(nn.Module):
    def __init__(self, num_classes=1, base_ch=32, base_depth=1,
                 operator="concat", fuse_stages=(3, 4, 5)):
        super().__init__()
        # two independent backbones: the cameras are too different to share filters.
        self.backbone_rgb = CSPDarknet(base_ch=base_ch, base_depth=base_depth)
        self.backbone_thermal = CSPDarknet(base_ch=base_ch, base_depth=base_depth)

        channels = (base_ch * 4, base_ch * 8, base_ch * 16)   # dark3, dark4, dark5
        self.fuse_stages = set(fuse_stages)
        op = OPERATORS[operator]
        # one fusion module per scale; fused stages use the chosen operator, others sum.
        self.fusers = nn.ModuleList([
            op(ch) if stage in self.fuse_stages else SumFuse()
            for stage, ch in zip(_STAGES, channels)
        ])

        self.neck = PAFPN(base_ch=base_ch, base_depth=base_depth)
        self.head = DecoupledHead(num_classes=num_classes, in_channels=channels)

    def forward(self, rgb, thermal):
        feats_rgb = self.backbone_rgb(rgb)            # (d3, d4, d5)
        feats_thm = self.backbone_thermal(thermal)
        fused = tuple(fuser(r, t) for fuser, r, t in zip(self.fusers, feats_rgb, feats_thm))
        return self.head(self.neck(fused))
