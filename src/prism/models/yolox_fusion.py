"""QUARANTINED legacy two-stream model -- do NOT train new experiments with it.

This class implements the old `fuse_stages` behavior that the runbook (TODO.md
6.7) declares invalid as a depth sweep:

    Z_3 = R_3 + T_3          # unfused level 3 -> SUM
    Z_4 = Conv1x1([R_4;T_4]) # fused level 4 -> CONCAT
    Z_5 = R_5 + T_5          # unfused level 5 -> SUM

Every neck input contains BOTH modalities regardless of the declared config, and
the operator changes at dark4 while fusion persists at dark3/dark5. It does NOT
represent streams meeting at dark4.

- All results produced with this implementation (the Phase-1 F-cat-345 numbers)
  are quarantined: they may be reported as *mixed-operator multi-scale fusion*
  once the graph is verified, but they cannot support any meeting-depth claim.
- Kept under its original name as an alias so old checkpoints still load.

Replacement graphs: `prism.models.one_shot_fusion.OneShotFusionYOLOX` (true
one-shot depth sweep, TODO.md 6.3) and
`prism.models.multi_scale_fusion.MultiScaleFusionYOLOX` (explicit anchor-only
fallback at unfused levels, TODO.md 6.4).
"""

from __future__ import annotations

from torch import nn

from prism.models.darknet import CSPDarknet
from prism.models.fusion import OPERATORS, SumFuse
from prism.models.head import DecoupledHead
from prism.models.pafpn import PAFPN

_STAGES = (3, 4, 5)


class LegacyTwoStreamYOLOX(nn.Module):
    """QUARANTINED -- see module docstring. New runs must use
    OneShotFusionYOLOX / MultiScaleFusionYOLOX."""

    def __init__(
        self, num_classes=1, base_ch=32, base_depth=1, operator="concat", fuse_stages=(3, 4, 5)
    ):
        super().__init__()
        self.backbone_rgb = CSPDarknet(base_ch=base_ch, base_depth=base_depth)
        self.backbone_thermal = CSPDarknet(base_ch=base_ch, base_depth=base_depth)

        channels = (base_ch * 4, base_ch * 8, base_ch * 16)  # dark3, dark4, dark5
        self.fuse_stages = set(fuse_stages)
        op = OPERATORS[operator]
        self.fusers = nn.ModuleList(
            [
                op(ch) if stage in self.fuse_stages else SumFuse()
                for stage, ch in zip(_STAGES, channels, strict=True)
            ]
        )

        self.neck = PAFPN(base_ch=base_ch, base_depth=base_depth)
        self.head = DecoupledHead(num_classes=num_classes, in_channels=channels)

    def forward(self, rgb, thermal):
        feats_rgb = self.backbone_rgb(rgb)
        feats_thm = self.backbone_thermal(thermal)
        fused = tuple(
            fuser(r, t) for fuser, r, t in zip(self.fusers, feats_rgb, feats_thm, strict=True)
        )
        return self.head(self.neck(fused))


# Alias kept so old checkpoints load unchanged; new code must not construct it.
TwoStreamYOLOX = LegacyTwoStreamYOLOX
