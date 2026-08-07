"""Multi-scale (multi-level) fusion: both complete backbones stay independent and
fuse at a declared subset of scales. This is the family that answers *which
feature scales should receive complementary information* -- not where the streams
become one (that is OneShotFusionYOLOX). TODO.md 6.4.

    R_3,R_4,R_5 = B_r(x_r)
    T_3,T_4,T_5 = B_t(x_t)
    Z_l = F_l(R_l, T_l)  if l in S
    Z_l = A_l(T_l)       if l not in S     [thermal anchor]
    y = H(N(Z_3, Z_4, Z_5))

Required subsets S: {3}, {4}, {5}, {3,4}, {4,5}, {3,4,5}.

Every level NOT in S is fed by a single modality through an explicit adapter --
never a silent sum. That is the fix for the invalid `fuse_stages` behavior
(TODO.md 6.7) where un-fused levels were summed, so every neck input contained
both modalities regardless of the declared config.
"""

from __future__ import annotations

from torch import nn

from prism.models.darknet import CSPDarknet
from prism.models.fusion import AdapterProjection, build_fusion_operator
from prism.models.head import DecoupledHead
from prism.models.pafpn import PAFPN

_NECK_LEVELS = (3, 4, 5)


class MultiScaleFusionYOLOX(nn.Module):
    def __init__(
        self,
        num_classes=1,
        base_ch=32,
        base_depth=1,
        fuse_subset=(3, 4, 5),
        anchor="thermal",
        operator="concat",
    ):
        if anchor not in ("thermal", "rgb"):
            raise ValueError(f"anchor must be 'thermal' or 'rgb', got {anchor}")
        subset = set(fuse_subset)
        if not subset <= set(_NECK_LEVELS) or not subset:
            raise ValueError(f"fuse_subset must be a non-empty subset of {_NECK_LEVELS}")
        super().__init__()
        self.fuse_subset = sorted(subset)
        self.anchor = anchor
        self.level_channels = {3: base_ch * 4, 4: base_ch * 8, 5: base_ch * 16}

        self.backbone_rgb = CSPDarknet(base_ch=base_ch, base_depth=base_depth)
        self.backbone_thermal = CSPDarknet(base_ch=base_ch, base_depth=base_depth)

        self.fusers = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        for lvl in _NECK_LEVELS:
            if lvl in subset:
                self.fusers[str(lvl)] = build_fusion_operator(
                    operator, self.level_channels[lvl], anchor=anchor
                )
            else:
                self.adapters[str(lvl)] = AdapterProjection(
                    self.level_channels[lvl], self.level_channels[lvl]
                )

        self.neck = PAFPN(base_ch=base_ch, base_depth=base_depth)
        self.head = DecoupledHead(
            num_classes=num_classes,
            in_channels=tuple(self.level_channels[level] for level in _NECK_LEVELS),
        )

    def fusion_levels(self):
        return tuple(self.fuse_subset)

    def forward_features(self, rgb, thermal):
        r3, r4, r5 = self.backbone_rgb(rgb)
        t3, t4, t5 = self.backbone_thermal(thermal)
        r = {3: r3, 4: r4, 5: r5}
        t = {3: t3, 4: t4, 5: t5}
        out = {}
        for lvl in _NECK_LEVELS:
            if str(lvl) in self.fusers:
                out[lvl] = self.fusers[str(lvl)](r[lvl], t[lvl])
            elif self.anchor == "thermal":
                out[lvl] = self.adapters[str(lvl)](t[lvl])
            else:
                out[lvl] = self.adapters[str(lvl)](r[lvl])
        return out[3], out[4], out[5]

    def forward(self, rgb, thermal):
        return self.head(self.neck(self.forward_features(rgb, thermal)))

    def graph_table(self) -> str:
        rows = []
        for lvl in _NECK_LEVELS:
            ch = self.level_channels[lvl]
            stride = 2**lvl
            owner = "fused" if str(lvl) in self.fusers else self.anchor
            rows.append(f"  p{lvl}: C={ch}, stride={stride}, owned by [{owner}]")
        return "\n".join(rows)
