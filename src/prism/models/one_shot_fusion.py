"""True one-shot fusion: streams meet ONCE, at one declared backbone level, and
never split again. This is the literal reading of "one-shot middle fusion"
(TODO.md 6.3), replacing the invalid `fuse_stages` depth sweep (TODO.md 6.7).

Graph (thermal anchor, meet at dark4):

    R_1:4 -> dark4_r; T_1:4 -> dark4_t
    Z_4 = F_4(R_4, T_4)
    Z_5 = dark5_shared(Z_4)            # shared suffix, never splits again
    neck inputs = [A_3(T_3), Z_4, Z_5] # dark3 skip is thermal-only: RGB first
                                       # enters at dark4 -- the depth claim is literal.

Meet at dark3:   neck inputs = [Z_3, Z_4, Z_5] (all fused/shared)
Meet at dark5:   neck inputs = [A_3(T_3), A_4(T_4), Z_5]

The shared suffix stages are module-copies of the backbone's dark4/dark5 blocks,
NOT the rgb/thermal backbone instances -- after F_L exactly one stream exists.
`anchor` selects which modality feeds the un-fused neck skips (thermal default;
RGB anchor repeats the critical result, TODO.md 6.3).
"""

from __future__ import annotations

from torch import nn

from prism.models.darknet import CSPDarknet
from prism.models.fusion import AdapterProjection, build_fusion_operator
from prism.models.head import DecoupledHead
from prism.models.pafpn import PAFPN

_NECK_LEVELS = (3, 4, 5)


class OneShotFusionYOLOX(nn.Module):
    def __init__(
        self,
        num_classes=1,
        base_ch=32,
        base_depth=1,
        meet_level=4,
        anchor="thermal",
        operator="concat",
    ):
        if meet_level not in (3, 4, 5):
            raise ValueError(f"meet_level must be 3, 4 or 5, got {meet_level}")
        if anchor not in ("thermal", "rgb"):
            raise ValueError(f"anchor must be 'thermal' or 'rgb', got {anchor}")
        super().__init__()
        self.meet_level = meet_level
        self.anchor = anchor
        self.level_channels = {3: base_ch * 4, 4: base_ch * 8, 5: base_ch * 16}

        self.backbone_rgb = CSPDarknet(base_ch=base_ch, base_depth=base_depth)
        self.backbone_thermal = CSPDarknet(base_ch=base_ch, base_depth=base_depth)

        # one fusion call, at the meeting level only
        self.fuser = build_fusion_operator(operator, self.level_channels[meet_level], anchor=anchor)
        self.fusion_calls = 0  # incremented per forward; test hook target

        # shared suffix: dark_{L+1}..dark5 as fresh module copies
        b = CSPDarknet(base_ch=base_ch, base_depth=base_depth)
        stages = [b.dark3, b.dark4, b.dark5]  # index 0 -> dark3
        self.suffix_stages = (
            nn.ModuleList(stages[meet_level - 2 :]) if meet_level < 5 else nn.ModuleList()
        )

        # adapters for the un-fused neck skips (levels < meet_level)
        self.adapters = nn.ModuleDict()
        for lvl in _NECK_LEVELS:
            if lvl < meet_level:
                self.adapters[str(lvl)] = AdapterProjection(
                    self.level_channels[lvl], self.level_channels[lvl]
                )

        self.neck = PAFPN(base_ch=base_ch, base_depth=base_depth)
        self.head = DecoupledHead(
            num_classes=num_classes,
            in_channels=tuple(self.level_channels[level] for level in _NECK_LEVELS),
        )

    def forward_features(self, rgb, thermal):
        """Return the three neck inputs (p3, p4, p5) as the graph defines them."""
        L = self.meet_level
        # the anchor branch also supplies the un-fused skips below L
        # Both private streams end at L. forward_upto(L) already contains every
        # lower feature needed for anchor-only neck skips.
        r = self.backbone_rgb.forward_upto(rgb, L)
        t = self.backbone_thermal.forward_upto(thermal, L)

        z = self.fuser(r[L], t[L])
        self.fusion_calls += 1
        out = {L: z}
        cur = z
        for stage, lvl in zip(self.suffix_stages, range(L + 1, 6), strict=True):
            cur = stage(cur)
            out[lvl] = cur

        inputs = []
        for lvl in _NECK_LEVELS:
            if lvl >= L:
                inputs.append(out[lvl])
            elif self.anchor == "thermal":
                inputs.append(self.adapters[str(lvl)](t[lvl]))
            else:
                inputs.append(self.adapters[str(lvl)](r[lvl]))
        return tuple(inputs)

    def forward(self, rgb, thermal):
        return self.head(self.neck(self.forward_features(rgb, thermal)))

    def graph_table(self) -> str:
        """Human-readable graph spec (TODO.md 6.0: publish input/output shapes,
        stride, channels, and per-tensor modality ownership)."""
        rows = []
        for lvl in _NECK_LEVELS:
            ch = self.level_channels[lvl]
            stride = 2**lvl
            owner = "fused" if lvl >= self.meet_level else self.anchor
            rows.append(f"  p{lvl}: C={ch}, stride={stride}, owned by [{owner}]")
        return "\n".join(rows)
